import os
import argparse
import json
import time
import requests
import hmac
import hashlib
import base64
from dotenv import load_dotenv

load_dotenv()

# --- ACRCLOUD CONFIG ---
ACR_HOST = "identify-eu-west-1.acrcloud.com"
ACR_ACCESS_KEY = os.getenv("ACR_ACCESS_KEY")
ACR_ACCESS_SECRET = os.getenv("ACR_ACCESS_SECRET")
OUTPUT_BASE = "output_assets"

def get_acr_signature(http_method, http_uri, access_key, access_secret, timestamp):
    string_to_sign = f"{http_method}\n{http_uri}\n{access_key}\n{timestamp}\n1"
    return base64.b64encode(hmac.new(access_secret.encode('ascii'), string_to_sign.encode('ascii'), digestmod=hashlib.sha1).digest()).decode('ascii')

def identify_music(file_path):
    if not ACR_ACCESS_KEY or not ACR_ACCESS_SECRET:
        return {"error": "Missing ACRCloud API Keys in .env"}

    request_url = f"http://{ACR_HOST}/v1/identify"
    timestamp = str(int(time.time()))
    signature = get_acr_signature("POST", "/v1/identify", ACR_ACCESS_KEY, ACR_ACCESS_SECRET, timestamp)

    files = {'sample': open(file_path, 'rb')}
    data = {
        'access_key': ACR_ACCESS_KEY,
        'sample_bytes': os.path.getsize(file_path),
        'timestamp': timestamp,
        'signature': signature,
        'data_type': 'audio',
        "signature_version": "1"
    }

    try:
        response = requests.post(request_url, files=files, data=data)
        result = response.json()

        if result['status']['code'] == 0:
            metadata = result['metadata']
            music_list = metadata.get('music', [])
            tracks = []
            for music in music_list:
                tracks.append({
                    "title": music.get('title'),
                    "artist": ", ".join([a['name'] for a in music.get('artists', [])]),
                    "album": music.get('album', {}).get('name'),
                    "score": music.get('score'),
                    "play_offset_ms": music.get('play_offset_ms'),
                    "duration_ms": music.get('duration_ms')
                })

            return {
                "music_present": True,
                "track_count": len(tracks),
                "tracks": tracks
            }
        else:
            return {
                "music_present": False,
                "status": result['status']['msg']
            }

    except Exception as e:
        return {"error": str(e), "music_present": False}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_id')
    args = parser.parse_args()

    # 1. Determine Input File (Prefer Instrumental Stem)
    stem_path = f"storage/stems/{args.video_id}_no_vocals.mp3"
    raw_path = f"storage/audio/{args.video_id}.mp3"

    target_file = stem_path if os.path.exists(stem_path) else raw_path

    if not os.path.exists(target_file):
        print(f"[Error] Audio file not found: {target_file}")
        return

    print(f"Identifying music in: {target_file}...")
    result = identify_music(target_file)

    # 2. Save Result
    out_dir = os.path.join(OUTPUT_BASE, args.video_id)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{args.video_id}_music.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Music Analysis Saved: {out_path}")

if __name__ == "__main__":
    main()
