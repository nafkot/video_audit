from openai import OpenAI
import os
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI()

audio_filename = "sOsqXKr4l30.mp3"

if not os.path.exists(audio_filename):
    print(f"Error: File '{audio_filename}' not found.")
    exit()

print(f"Uploading {audio_filename} ({os.path.getsize(audio_filename)/1024/1024:.2f} MB) to OpenAI...")

# Start the timer
start_time = time.time()

try:
    audio_file = open(audio_filename, "rb")

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json",
        timestamp_granularities=["segment"]
    )

    # Stop the timer
    end_time = time.time()
    processing_time = end_time - start_time

    print("\n--- Transcription Success! ---")
    print(f"Audio Length: {transcript.duration:.2f} seconds")
    print(f"Actual Processing Time: {processing_time:.2f} seconds")

    # --- SAVE TO FILE ---
    output_filename = "transcript.json"

    # Convert the object to a dictionary for saving
    # We reconstruct the dictionary because the response object isn't directly serializable
    data_to_save = {
        "text": transcript.text,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            }
            for seg in transcript.segments
        ]
    }

    with open(output_filename, "w") as f:
        json.dump(data_to_save, f, indent=2)

    print(f"\nFull transcript saved to: {output_filename}")

    print("\nSample Segments:")
    for segment in transcript.segments[:3]:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s]: {segment.text}")

except Exception as e:
    print(f"\nError: {e}")
