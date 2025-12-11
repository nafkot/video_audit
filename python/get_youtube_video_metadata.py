#!/usr/bin/env python3
import os
import argparse
import json
from datetime import datetime, timezone
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Config
YOUTUBE_SERVICE_ACCOUNT_FILE = 'account.json'
YOUTUBE_SCOPE = ['https://www.googleapis.com/auth/youtube.readonly']
OUTPUT_BASE = "output_assets"

def get_service():
    if not os.path.exists(YOUTUBE_SERVICE_ACCOUNT_FILE):
        print(f"Error: {YOUTUBE_SERVICE_ACCOUNT_FILE} not found.")
        return None
    creds = service_account.Credentials.from_service_account_file(
        YOUTUBE_SERVICE_ACCOUNT_FILE, scopes=YOUTUBE_SCOPE
    )
    return build('youtube', 'v3', credentials=creds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    args = parser.parse_args()

    youtube = get_service()
    if not youtube: return

    # 1. Video Request
    print(f"Fetching metadata for {args.id}...")
    vid_response = youtube.videos().list(
        part='snippet,statistics,contentDetails',
        id=args.id
    ).execute()

    if not vid_response['items']:
        print("Video not found.")
        return

    video = vid_response['items'][0]
    channel_id = video['snippet']['channelId']

    # 2. Channel Request
    ch_response = youtube.channels().list(
        part='snippet,statistics,contentDetails',
        id=channel_id
    ).execute()

    channel = ch_response['items'][0]

    # 3. Calculate Frequency & Age
    pub_date = channel['snippet']['publishedAt']
    try:
        created_dt = datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except:
        created_dt = datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    days_active = (datetime.now(timezone.utc) - created_dt).days or 1
    total_videos = int(channel['statistics']['videoCount'])

    freq_week = round((total_videos / days_active) * 7, 2)
    freq_month = round((total_videos / days_active) * 30, 2)

    # 4. Construct Data Package
    data = {
        "video": {
            "id": args.id,
            "title": video['snippet']['title'],
            "description": video['snippet']['description'],
            "created_at": video['snippet']['publishedAt'],
            "views": int(video['statistics'].get('viewCount', 0)),
            "likes": int(video['statistics'].get('likeCount', 0)),
            "duration": video['contentDetails']['duration']
        },
        "channel": {
            "id": channel_id,
            "name": channel['snippet']['title'],
            "subscribers": int(channel['statistics']['subscriberCount']),
            "total_videos": total_videos,
            "created_at": pub_date,
            "upload_freq": {
                "per_week": freq_week,
                "per_month": freq_month
            }
        }
    }

    # 5. Save
    out_dir = os.path.join(OUTPUT_BASE, args.id)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, f"{args.id}_details.json"), 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Metadata saved to {out_dir}")

if __name__ == '__main__':
    main()
