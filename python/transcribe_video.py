import os
import argparse
import json
import sys

# Add parent directory to path to import ingestion modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.transcript import get_transcript_segments

OUTPUT_BASE = "output_assets"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    args = parser.parse_args()

    print(f"Fetching transcript for {args.id}...")
    segments = get_transcript_segments(args.id)

    out_dir = os.path.join(OUTPUT_BASE, args.id)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{args.id}_transcription.json")

    # Save as LIST []
    with open(out_path, 'w') as f:
        json.dump(segments if segments else [], f, indent=2)

    print(f"Transcript saved to {out_path}")

if __name__ == "__main__":
    main()
