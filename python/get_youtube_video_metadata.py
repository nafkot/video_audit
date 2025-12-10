#!/usr/bin/env python3
import os
import argparse
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Config
YOUTUBE_SERVICE_ACCOUNT_FILE = 'account.json'
YOUTUBE_SCOPE = ['https://www.googleapis.com/auth/youtube.readonly']
OUTPUT_BASE = "output_assets"

def get_authenticated_service():
    if not os.path.exists(YOUTUBE_SERVICE_ACCOUNT_FILE):
        print("Error: account.json not found in root folder.")
        return None
    creds = service_account.Credentials.from_service_account_file(
        YOUTUBE_SERVICE_ACCOUNT_FILE, scopes=YOUTUBE_SCOPE
    )
    return build('youtube', 'v3', credentials=creds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    args = parser.parse_args()

    youtube = get_authenticated_service()
    if not youtube: return

    try:
        request = youtube.videos().list(
            part='snippet,statistics,contentDetails',
            id=args.id
        )
        response = request.execute()

        if not response['items']:
            print("Video not found.")
            return

        vid = response['items'][0]
        data = {
            "title": vid['snippet']['title'],
            "channel_title": vid['snippet']['channelTitle'],
            "description": vid['snippet']['description'],
            "view_count": int(vid['statistics'].get('viewCount', 0)),
            "like_count": int(vid['statistics'].get('likeCount', 0)),
            "published_at": vid['snippet']['publishedAt']
        }

        # --- SAVE TO FILE (Critical Fix) ---
        out_dir = os.path.join(OUTPUT_BASE, args.id)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{args.id}_details.json")
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Metadata saved to {out_path}")

    except Exception as e:
        print(f"API Error: {e}")

if __name__ == '__main__':
    main()
