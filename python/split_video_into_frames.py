#!/usr/bin/env python3

import os
import argparse
import json
import dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

dotenv.load_dotenv()

DEBUG = os.getenv('DEBUG')
CONTENT_PATH = os.getenv('CONTENT_PATH')

def main():
    args = parse_arguments()
    videos_base_path = f"{CONTENT_PATH}/videos"
    fps = args.fps

    print(f"videos_base_path: {videos_base_path}");
    if not os.path.exists(videos_base_path):
        return print(json.dumps(False))

    os.makedirs(f"{CONTENT_PATH}/videos/frames", exist_ok=True)

    files = [f for f in args.files.split(',') if os.path.exists(f"{videos_base_path}/{f}")]

    results = []

    with ThreadPoolExecutor(max_workers=(os.cpu_count() * 10)) as executor:
        futures = [executor.submit(split_video_into_frames, f"{videos_base_path}/{file}", fps) for file in files]
        for future in as_completed(futures):
            results.append(future.result())
            pass

    print(json.dumps(results, indent=2))

def parse_arguments():
    parser = argparse.ArgumentParser(description='split video files into frames')
    parser.add_argument('--files', required=True, help='name of the video files to split')
    parser.add_argument('--fps', default='0.5', help='name of the video files to split')
    return parser.parse_args()

def split_video_into_frames(filepath, fps=0.5):
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_path = f"{CONTENT_PATH}/videos/frames/{base_name}/%04d.png"
    output_dir = f"{CONTENT_PATH}/videos/frames/{base_name}/"

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(output_dir):
        print(f"filename: {filename}");
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                if DEBUG: print(f"Failed to remove {file_path}: {e}")


    command = [
        "ffmpeg",
        "-i", filepath,
        "-vf", f"fps={fps}",  # extract # frames per second
        output_path
    ]

    response = {
        'file': base_name,
        'success': False,
    }

    try:
        subprocess.run(command, check=True)
        response['success'] = True
        if DEBUG: print("Frames extracted successfully.")
    except subprocess.CalledProcessError as e:
        if DEBUG: print(f"Error during extraction: {e}")

    return response

if __name__ == '__main__':
    main()
