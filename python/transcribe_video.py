import os
import argparse
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

OUTPUT_BASE = "output_assets"
STORAGE_AUDIO = "storage/audio"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True, help="Video ID to transcribe")
    args = parser.parse_args()

    video_id = args.id
    print(f"--- Transcribing Vocals for {video_id} ---")

    # 1. Locate the Vocals file (from LALAL.ai step)
    # Fallback to main audio if vocals missing (e.g. if LALAL failed/skipped)
    vocab_path = os.path.join(STORAGE_AUDIO, f"{video_id}_vocals.mp3")
    main_audio_path = os.path.join(STORAGE_AUDIO, f"{video_id}.mp3")

    target_file = vocab_path if os.path.exists(vocab_path) else main_audio_path

    if not os.path.exists(target_file):
        print(f"❌ Error: Audio file not found at {target_file}")
        exit(1)

    print(f"🎤 Using audio source: {target_file}")

    # 2. Initialize OpenAI
    client = OpenAI()

    try:
        # 3. Transcribe with Whisper-1
        with open(target_file, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        # 4. Prepare Segments Data
        # Ensure we have a list of segments [ {start, end, text}, ... ]
        segments = []
        if hasattr(transcript, 'segments'):
            segments = transcript.segments
        elif isinstance(transcript, dict):
            segments = transcript.get('segments', [])

        # 5. Save Output
        out_dir = os.path.join(OUTPUT_BASE, video_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{video_id}_transcription.json")

        # Convert objects to dict if necessary for JSON dump
        serializable_segments = []
        for seg in segments:
            item = seg if isinstance(seg, dict) else seg.model_dump()
            serializable_segments.append({
                "start": item.get('start'),
                "end": item.get('end'),
                "text": item.get('text', '').strip()
            })

        with open(out_path, 'w') as f:
            json.dump(serializable_segments, f, indent=2)

        print(f"✅ Transcript saved to {out_path}")

    except Exception as e:
        print(f"❌ OpenAI Transcription Failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
