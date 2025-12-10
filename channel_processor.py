import os
import argparse
import json
import sqlite3
import yt_dlp
import time
import re
import requests
import html
from datetime import datetime
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from openai import OpenAI
from httpx import Timeout

# --- CONFIGURATION ---
load_dotenv()
YOUTUBE_SERVICE_ACCOUNT_FILE = 'account.json'
YOUTUBE_SCOPE = ['https://www.googleapis.com/auth/youtube.readonly']
DB_NAME = 'youtube_insights.db'
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
RAPIDAPI_HOST = "youtube-captions-transcript-subtitles-video-combiner.p.rapidapi.com"

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=Timeout(300.0, connect=10.0)
)

# --- PART 1: RAPID API HANDLERS ---

def fetch_rapidapi_details(video_id):
    cache_file = f"{video_id}_details.json"
    if os.path.exists(cache_file):
        print(f"   [{video_id}] Found cached details JSON.")
        with open(cache_file, 'r') as f: return json.load(f)

    print(f"   [{video_id}] Fetching details from RapidAPI...")
    url = f"https://{RAPIDAPI_HOST}/get-video-info/{video_id}"
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": RAPIDAPI_HOST}
    try:
        response = requests.get(url, headers=headers, params={"format": "json"}, timeout=15)
        if response.status_code == 200:
            data = response.json()
            with open(cache_file, 'w') as f: json.dump(data, f, indent=2)
            return data
        return None
    except Exception as e:
        print(f"   [Error] RapidAPI Details: {e}")
        return None

def fetch_rapidapi_transcript(video_id):
    cache_file = f"{video_id}_transcription.json"
    if os.path.exists(cache_file):
        print(f"   [{video_id}] Found cached transcript JSON.")
        with open(cache_file, 'r') as f: return json.load(f)

    print(f"   [{video_id}] Fetching transcript from RapidAPI...")
    url = f"https://{RAPIDAPI_HOST}/download-all/{video_id}"
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": RAPIDAPI_HOST}
    try:
        response = requests.get(url, headers=headers, params={"format_subtitle": "json"}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            with open(cache_file, 'w') as f: json.dump(data, f, indent=2)
            return data
        return None
    except Exception as e:
        print(f"   [Error] RapidAPI Transcript: {e}")
        return None

def normalize_rapidapi_transcript(rapid_data):
    """
    Converts RapidAPI subtitle lines into larger, meaningful chunks (~45-60s).
    This solves the 'fragmented context' issue in search.
    """
    if not rapid_data: return None

    segments = []
    full_text = ""

    try:
        # RapidAPI returns list of dicts, typically index 0 has subtitles
        subtitle_data = rapid_data[0].get('subtitle', [])

        current_chunk = {"text": "", "start": 0.0, "end": 0.0}

        for i, item in enumerate(subtitle_data):
            start = float(item['start'])
            dur = float(item['dur'])
            text = html.unescape(item['text']).strip()

            # Initialize chunk start time if new
            if current_chunk["text"] == "":
                current_chunk["start"] = start

            # Append text
            current_chunk["text"] += text + " "
            current_chunk["end"] = start + dur

            # CHUNKING LOGIC:
            # Create a new segment if:
            # 1. Current chunk is > 45 seconds OR
            # 2. We hit a sentence end (.?!) AND chunk is > 20 seconds
            is_long_enough = (current_chunk["end"] - current_chunk["start"]) > 45
            is_sentence_end = text.endswith(('.', '!', '?')) and (current_chunk["end"] - current_chunk["start"]) > 20

            if is_long_enough or is_sentence_end or i == len(subtitle_data) - 1:
                segments.append({
                    'start': current_chunk["start"],
                    'end': current_chunk["end"],
                    'text': current_chunk["text"].strip()
                })
                full_text += current_chunk["text"]
                # Reset
                current_chunk = {"text": "", "start": 0.0, "end": 0.0}

        class MockTranscript:
            def __init__(self, text, segs):
                self.text = text
                self.segments = segs
        return MockTranscript(full_text, segments)

    except Exception as e:
        print(f"   [Error] Parsing RapidAPI JSON: {e}")
        return None

# --- PART 2: YOUTUBE API ---
def get_authenticated_service():
    if not os.path.exists(YOUTUBE_SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(f"Missing {YOUTUBE_SERVICE_ACCOUNT_FILE}")
    credentials = service_account.Credentials.from_service_account_file(
        YOUTUBE_SERVICE_ACCOUNT_FILE, scopes=YOUTUBE_SCOPE
    )
    return build('youtube', 'v3', credentials=credentials)

def get_channel_metadata_api(youtube, channel_id):
    try:
        request = youtube.channels().list(part='snippet,statistics,brandingSettings,contentDetails', id=channel_id)
        response = request.execute()
        if not response['items']: return None
        info = response['items'][0]
        return {
            'id': info['id'],
            'title': info['snippet']['title'],
            'description': info['snippet']['description'],
            'subscribers': int(info['statistics']['subscriberCount']),
            'total_videos': int(info['statistics']['videoCount']),
            'views': int(info['statistics']['viewCount']),
            'created': info['snippet']['publishedAt'][:10],
            'thumbnail': info['snippet']['thumbnails']['high']['url'],
            'uploads_id': info['contentDetails']['relatedPlaylists']['uploads']
        }
    except Exception as e:
        print(f"Error fetching channel metadata: {e}")
        return None

def get_latest_video_ids(youtube, playlist_id, limit=10):
    request = youtube.playlistItems().list(part='contentDetails', playlistId=playlist_id, maxResults=limit)
    response = request.execute()
    return [item['contentDetails']['videoId'] for item in response['items']]

# --- PART 3: FALLBACK PIPELINE ---
def download_audio_fallback(video_id):
    filename = f"{video_id}.mp3"
    if os.path.exists(filename): return filename
    print(f"   [Fallback] Attempting download for {video_id}...")
    formats = ['best', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4', 'bestaudio/best']
    for fmt in formats:
        try:
            ydl_opts = {
                'format': fmt, 'outtmpl': video_id, 'quiet': True, 'cookiefile': 'cookies.txt',
                'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '32'}],
                'postprocessor_args': ['-ac', '1'], 'nocheckcertificate': True,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            if os.path.exists(filename): return filename
        except Exception: continue
    return None

def transcribe_audio_fallback(audio_filename):
    if not audio_filename or not os.path.exists(audio_filename): return None
    print(f"   [Fallback] Whisper Transcribing...")
    try:
        audio_file = open(audio_filename, "rb")
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="verbose_json", timestamp_granularities=["segment"]
        )
        return transcript
    except Exception: return None

# --- PART 4: ANALYSIS & DB ---
def analyze_transcript(text, video_id):
    print(f"   [{video_id}] Analyzing content...")
    if not text: return None
    clean_text = html.unescape(text)
    prompt = f"""
    Analyze this video transcript.
    TASKS:
    1. Write a 3-sentence summary.
    2. Determine sentiment (Positive/Neutral/Negative).
    3. Extract 5 topics (comma-separated).
    4. COMMERCIAL INTELLIGENCE: List EVERY company/brand mentioned (EXHAUSTIVE).
    5. Extract Sponsors.
    Output JSON: {{ "summary": "...", "sentiment": "...", "topics": "...", "brands": [], "sponsors": [] }}
    Transcript: {clean_text[:15000]}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], response_format={ "type": "json_object" }, temperature=0.3
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"   [Error] AI Analysis failed: {e}")
        return None

def save_channel_profile(meta):
    print(f"   [DB] Saving Channel Profile: {meta['title']}...")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO channels (channel_id, title, description, subscriber_count, video_count, view_count, creation_date, thumbnail_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (meta['id'], meta['title'], meta['description'], meta['subscribers'], meta['total_videos'], meta['views'], meta['created'], meta['thumbnail']))
    conn.commit()
    conn.close()

def save_video_to_db(video_meta, analysis, transcript):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        brands_json = json.dumps(analysis.get('brands', []))
        sponsors_json = json.dumps(analysis.get('sponsors', []))
        c.execute('''INSERT OR REPLACE INTO videos (video_id, channel_id, title, channel_name, upload_date, duration, overall_summary, overall_sentiment, topics, brands, sponsors, view_count, like_count, thumbnail_url, author, is_family_safe, owner_profile_url, category) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
            video_meta['id'], video_meta['channel_id'], video_meta['title'], video_meta['channel'], video_meta['date'], video_meta['duration'],
            analysis.get('summary', ''), analysis.get('sentiment', 'Neutral'), analysis.get('topics', ''),
            brands_json, sponsors_json, video_meta.get('viewCount', 0), video_meta.get('likeCount', 0), video_meta.get('thumbnail', ''),
            video_meta.get('author', ''), video_meta.get('isFamilySafe', 1), video_meta.get('ownerProfileUrl', ''), video_meta.get('category', '')
        ))

        c.execute('DELETE FROM video_segments WHERE video_id = ?', (video_meta['id'],))

        segments = getattr(transcript, 'segments', [])
        data_tuples = []
        for seg in segments:
            # Safe access
            s_start = float(seg.get('start')) if isinstance(seg, dict) else seg.start
            s_end = float(seg.get('end')) if isinstance(seg, dict) else seg.end
            s_text = seg.get('text') if isinstance(seg, dict) else seg.text
            data_tuples.append((video_meta['id'], s_start, s_end, s_text))

        c.executemany('INSERT INTO video_segments (video_id, start_time, end_time, text) VALUES (?, ?, ?, ?)', data_tuples)
        conn.commit()
        print(f"   [{video_meta['id']}] Saved to DB successfully.")
    except Exception as e:
        print(f"   [{video_meta['id']}] [Error] DB Save Failed: {e}")
    finally:
        conn.close()

def is_video_processed(video_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT 1 FROM videos WHERE video_id = ?", (video_id,))
    res = c.fetchone()
    conn.close()
    return res is not None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel_id', required=True)
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()

    print(f"--- Processing Channel: {args.channel_id} ---")
    try:
        youtube = get_authenticated_service()
        channel_meta = get_channel_metadata_api(youtube, args.channel_id)
        if not channel_meta: return
        save_channel_profile(channel_meta)
        video_ids = get_latest_video_ids(youtube, channel_meta['uploads_id'], args.limit)
    except Exception as e:
        print(f"Setup Error: {e}")
        return

    for vid in video_ids:
        print(f"\n>>> Checking: {vid}")
        if is_video_processed(vid):
            print(f"   [{vid}] Already processed. Skipping.")
            continue

        rapid_details = fetch_rapidapi_details(vid)
        video_meta = {'id': vid, 'channel_id': args.channel_id, 'channel': channel_meta['title'], 'title': 'Unknown', 'date': '', 'duration': 0}

        if rapid_details:
            if rapid_details.get('isShortsEligible', False):
                print(f"   [{vid}] Skipped (Shorts Detected)")
                continue

            thumb = rapid_details.get('thumbnail', [{}])[0].get('url', '')
            video_meta.update({
                'title': rapid_details.get('title'),
                'date': rapid_details.get('uploadDate'),
                'duration': int(rapid_details.get('lengthSeconds', 0) or 0),
                'viewCount': int(rapid_details.get('viewCount', 0) or 0),
                'likeCount': int(rapid_details.get('likeCount', 0) or 0),
                'thumbnail': thumb,
                'author': rapid_details.get('author'),
                'isFamilySafe': rapid_details.get('isFamilySafe'),
                'ownerProfileUrl': rapid_details.get('ownerProfileUrl'),
                'category': rapid_details.get('category'),
            })

            if video_meta['duration'] < 60:
                print(f"   [{vid}] Skipped (Duration < 60s)")
                continue

        # TRANSCRIPT
        transcript = None
        rapid_transcript_json = fetch_rapidapi_transcript(vid)

        if rapid_transcript_json:
            transcript = normalize_rapidapi_transcript(rapid_transcript_json)

        if not transcript:
            print(f"   [{vid}] Fallback to Whisper...")
            audio_file = download_audio_fallback(vid)
            transcript = transcribe_audio_fallback(audio_file)
            if audio_file and os.path.exists(audio_file): os.remove(audio_file)

        if not transcript:
            print(f"   [{vid}] Skipped (No transcript)")
            continue

        full_text = getattr(transcript, 'text', '')
        analysis = analyze_transcript(full_text, vid)
        if analysis:
            save_video_to_db(video_meta, analysis, transcript)

    print("\n--- All jobs finished! ---")

if __name__ == '__main__':
    main()
