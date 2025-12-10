import sys
import json
import time
import requests
import openai

from google.oauth2 import service_account
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
YOUTUBE_SERVICE_ACCOUNT_FILE = "account.json"
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

MAX_VIDEOS = 10

# ----------------------------------------------------------
# 1. Build YouTube API client using Service Account
# ----------------------------------------------------------
def build_youtube_client():
    credentials = service_account.Credentials.from_service_account_file(
        YOUTUBE_SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/youtube.readonly"]
    )

    youtube = build(
        "youtube",
        "v3",
        credentials=credentials
    )

    return youtube


# ----------------------------------------------------------
# 2. Fetch latest videos from channel
# ----------------------------------------------------------
def get_latest_videos(channel_id):
    youtube = build_youtube_client()

    print(f"\n🔍 Fetching latest videos for: {channel_id}")

    request = youtube.search().list(
        part="id",
        channelId=channel_id,
        maxResults=MAX_VIDEOS,
        order="date"
    )
    response = request.execute()

    video_ids = [
        item["id"]["videoId"]
        for item in response.get("items", [])
        if item["id"]["kind"] == "youtube#video"
    ]

    print(f"📌 Found {len(video_ids)} videos")
    return video_ids


# ----------------------------------------------------------
# 3A. Try free local transcript first
# ----------------------------------------------------------
def try_local_transcript(video_id):
    try:
        print("➡ Trying local youtube-transcript-api…")
        segments = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([s["text"] for s in segments])
        print("✔ Local transcript success")
        return full_text
    except:
        print("❌ No local transcript")
        return None


# ----------------------------------------------------------
# 3B. Use RapidAPI transcript as fallback
# ----------------------------------------------------------
def get_transcript_rapidapi(video_id):
    url = "https://youtube-transcripts-transcribe-youtube-video-to-text.p.rapidapi.com/transcribe"

    print("➡ Trying RapidAPI transcript…")

    payload = {"url": f"https://www.youtube.com/watch?v={video_id}"}

    headers = {
        "Content-Type": "application/json",
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "youtube-transcripts-transcribe-youtube-video-to-text.p.rapidapi.com"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        data = response.json()

        transcript = data.get("text")
        if transcript and transcript.strip():
            print("✔ RapidAPI transcript success")
            return transcript

        print(f"❌ RapidAPI returned no transcript: {data}")
        return None

    except Exception as e:
        print(f"❌ RapidAPI error: {e}")
        return None


# ----------------------------------------------------------
# 4. Analyse transcript with ChatGPT
# ----------------------------------------------------------
def analyse_transcript(video_id, transcript):
    print("\n🧠 Analysing transcript with ChatGPT…")

    prompt = f"""
You are an AI that analyses YouTube videos.

TRANSCRIPT:
{transcript}

Return valid JSON ONLY with:

- "topics": list of topics
- "sentiment": positive/neutral/negative
- "summary": 3–4 sentences
- "keywords": SEO keyword list
- "content_type": category (news, tutorial, commentary, documentary)
- "confidence_score": 1–100
"""

    try:
        result = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = result.choices[0].message["content"]
        return json.loads(content)

    except Exception as e:
        print("❌ ChatGPT parsing error:", e)
        return {"error": "Invalid JSON from ChatGPT"}


# ----------------------------------------------------------
# 5. Main pipeline
# ----------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_analysis.py CHANNEL_ID")
        sys.exit(1)

    channel_id = sys.argv[1]

    print("\n==============================")
    print("🎬 YOUTUBE ANALYSIS PIPELINE")
    print("==============================\n")

    video_ids = get_latest_videos(channel_id)
    results = []

    for video_id in video_ids:
        print(f"\n--------------------------------")
        print(f"🎥 Processing video: {video_id}")

        # Try local free transcript
        transcript = try_local_transcript(video_id)

        # Fallback to RapidAPI
        if not transcript:
            transcript = get_transcript_rapidapi(video_id)

        if not transcript:
            print("❌ No transcript found. Skipping video.")
            continue

        # Analyse transcript
        analysis = analyse_transcript(video_id, transcript)

        results.append({
            "video_id": video_id,
            "transcript_excerpt": transcript[:300],
            "analysis": analysis
        })

        time.sleep(1)

    # Save output
    file_name = f"analysis_{channel_id}.json"
    with open(file_name, "w") as f:
        json.dump(results, f, indent=2)

    print("\n🎉 DONE — Results saved to:", file_name, "\n")


if __name__ == "__main__":
    main()

