import os
import argparse
import json
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# You may need to adapt this if you are using Whisper locally vs RapidAPI
# This version assumes we are using the RapidAPI logic from ingestion/transcript.py
# BUT, since we have a local audio file now (vocals), RapidAPI might not be suitable 
# unless it accepts file uploads. 
# IF you are using RapidAPI (which fetches from YouTube ID), then passing --file is useless.
# IF you want to transcribe the LALAL stem, you need OpenAI Whisper or local Whisper.

# Assuming we revert to ID-based for RapidAPI, or implement local Whisper for the file.
# Let's support both to fix the crash.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    parser.add_argument('--file', help="Path to specific audio file (optional)", default=None)
    args = parser.parse_args()
    
    print(f"Fetching transcript for {args.id}...")
    
    data_to_save = []

    # Strategy: 
    # If using RapidAPI (External), we ignore --file and use --id.
    # If using Local Whisper, we use --file.
    
    # For now, let's stick to the working RapidAPI method (ID based) to fix the crash,
    # but acknowledge the file argument so argparse doesn't throw Exit Code 2.
    
    try:
        from ingestion.transcript import get_transcript_segments
        segments = get_transcript_segments(args.id)
        if segments:
            data_to_save = segments
            print(f"Retrieved {len(segments)} segments via API.")
        else:
            print("No transcript found via API.")
    except ImportError:
        print("Could not import ingestion module. Check paths.")
    
    # Save
    OUTPUT_BASE = "output_assets"
    out_dir = os.path.join(OUTPUT_BASE, args.id)
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, f"{args.id}_transcription.json")
    with open(out_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
        
    print(f"Transcript saved to {out_path}")

if __name__ == "__main__":
    main()
