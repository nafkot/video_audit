import os
import json
import sqlite3
import yt_dlp
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

DB_NAME = 'youtube_insights.db'

def get_video_metadata(video_url):
    ydl_opts = {'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return {
            'id': info['id'],
            'title': info['title'],
            'channel': info['uploader'],
            'date': info['upload_date'],
            'duration': info['duration']
        }

def transcribe_audio(video_id):
    # Check if we already have the transcript file to save time/money
    if os.path.exists("transcript.json"):
        print("Found existing transcript.json, using that.")
        with open("transcript.json", "r") as f:
            return json.load(f)

    # Otherwise, assume audio exists as mp3 (from your previous step)
    audio_filename = f"{video_id}.mp3"
    if not os.path.exists(audio_filename):
        print(f"Audio file {audio_filename} not found! Please download it first.")
        return None

    print("Transcribing with Whisper...")
    audio_file = open(audio_filename, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file, 
        response_format="verbose_json",
        timestamp_granularities=["segment"]
    )
    return transcript

def analyze_content(full_text):
    print("Analyzing with GPT-4o-mini...")
    prompt = f"""
    Analyze this video transcript.
    1. Write a 3-sentence summary.
    2. Determine the overall sentiment (Positive/Neutral/Negative).
    3. Extract 5 key topics as a comma-separated list.
    
    Output JSON format:
    {{
        "summary": "...",
        "sentiment": "...",
        "topics": "..."
    }}
    
    Transcript: {full_text[:15000]}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )
    return json.loads(response.choices[0].message.content)

def save_to_db(metadata, analysis, transcript_data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Insert Video
    c.execute('''
        INSERT OR REPLACE INTO videos (video_id, title, channel_name, upload_date, duration, overall_summary, overall_sentiment, topics)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        metadata['id'],
        metadata['title'],
        metadata['channel'],
        metadata['date'],
        metadata['duration'],
        analysis['summary'],
        analysis['sentiment'],
        analysis['topics']
    ))
    
    # Insert Segments (Chunks)
    # We group Whisper segments into larger chunks (e.g., ~60s) for better search results in the future
    # For now, we just save the raw segments
    
    # Clear old segments if re-ingesting
    c.execute('DELETE FROM video_segments WHERE video_id = ?', (metadata['id'],))
    
    segments = transcript_data.get('segments', [])
    for seg in segments:
        # handle object vs dict difference depending on source
        start = seg.get('start') if isinstance(seg, dict) else seg.start
        end = seg.get('end') if isinstance(seg, dict) else seg.end
        text = seg.get('text') if isinstance(seg, dict) else seg.text
        
        c.execute('''
            INSERT INTO video_segments (video_id, start_time, end_time, text)
            VALUES (?, ?, ?, ?)
        ''', (metadata['id'], start, end, text))
        
    conn.commit()
    conn.close()
    print(f"Successfully saved {metadata['title']} to database!")

def main():
    video_id = "sOsqXKr4l30" # Kurzgesagt Video
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    print(f"--- Processing {video_id} ---")
    
    # 1. Metadata
    meta = get_video_metadata(url)
    print(f"Title: {meta['title']}")
    
    # 2. Transcription
    # Note: Ensure you have the mp3 or transcript.json from previous steps
    transcript = transcribe_audio(video_id)
    if not transcript: return

    # Prepare text for AI
    # If transcript is an object (from API), use .text, else use dict key
    full_text = transcript.text if hasattr(transcript, 'text') else transcript.get('text', '')
    
    # 3. AI Analysis
    analysis = analyze_content(full_text)
    print("Analysis Complete.")
    
    # 4. Save
    save_to_db(meta, analysis, transcript)

if __name__ == "__main__":
    main()
