import subprocess
import csv
import json
import os
import re
import time
import sys
from datetime import datetime

# --- CONFIGURATION ---
VIDEO_SOURCE_DIR = "storage/videos"
SHELL_SCRIPT = "./python/analyse_video_360.sh"
OUTPUT_CSV = "batch_report.csv"

def get_video_files(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return []
    video_ids = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.mp4'):
            # Remove extension to get ID (e.g., 'abc.mp4' -> 'abc')
            vid_id = os.path.splitext(filename)[0]
            video_ids.append(vid_id)
    return sorted(video_ids)

def get_processed_ids(csv_file):
    if not os.path.exists(csv_file): return set()
    processed = set()
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row: processed.add(row[0])
    return processed

def extract_json_from_output(output_str):
    try:
        matches = list(re.finditer(r'\{.*\}', output_str, re.DOTALL))
        if not matches: return None
        return json.loads(matches[-1].group(0))
    except: return None

def append_to_csv(data_row):
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["VideoID", "Safety_Status", "Risk_Score", "Brands_Found", "Context_Summary", "Processed_At"])
        writer.writerow(data_row)

def main():
    print(f"--- Batch Processor Started ---")
    print(f"Scanning folder: {VIDEO_SOURCE_DIR}")

    all_videos = get_video_files(VIDEO_SOURCE_DIR)
    processed_videos = get_processed_ids(OUTPUT_CSV)
    videos_to_do = [v for v in all_videos if v not in processed_videos]

    print(f"Found {len(all_videos)} videos. Skipping {len(processed_videos)} done.")
    print(f"Queue: {len(videos_to_do)} videos.\n")

    for i, vid_id in enumerate(videos_to_do):
        print(f"\n{'='*60}")
        print(f"BATCH PROGRESS: [{i+1}/{len(videos_to_do)}] Processing: {vid_id}")
        print(f"{'='*60}\n")

        start_ts = time.time()
        full_output_log = ""

        # --- STREAMING EXECUTION ---
        try:
            # Popen allows us to read stdout line by line
            process = subprocess.Popen(
                [SHELL_SCRIPT, vid_id, "--service", "youtube"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge errors into output so we see them too
                text=True,
                encoding='utf-8',
                bufsize=1 # Line buffered
            )

            # Loop to print lines as they appear
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line, end='') # Print to your screen
                    full_output_log += line # Capture for JSON parsing later

            # 2. Parse Captured Data
            data = extract_json_from_output(full_output_log)

            if data:
                safety = data.get("safety", {})
                brands = data.get("brands", {})
                context = data.get("visual_context", [])

                brands_str = "; ".join([f"{k}" for k in brands.keys()]) if isinstance(brands, dict) else str(brands)
                context_str = " | ".join(context) if isinstance(context, list) else str(context)

                row = [
                    vid_id,
                    safety.get("status", "Unknown"),
                    f"{safety.get('score', 0):.2f}",
                    brands_str,
                    context_str,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ]
                append_to_csv(row)
                print(f"\n[+] Result Saved to CSV. (Time: {time.time()-start_ts:.1f}s)")
            else:
                print("\n[!] Error: Could not find JSON report in output.")
                append_to_csv([vid_id, "ERROR", "0.0", "JSON Parse Error", "", datetime.now()])

        except KeyboardInterrupt:
            print("\n\n[!] Batch stopped by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[!] System Error: {e}")

    print("\n--- Batch Complete ---")

if __name__ == "__main__":
    main()
