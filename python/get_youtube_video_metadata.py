#!/usr/bin/env python3

import os
import argparse
import json
from typing import Any
from datetime import datetime, timezone

import dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import mysql.connector

dotenv.load_dotenv()

# config
YOUTUBE_SERVICE_ACCOUNT_FILE = 'account.json'
YOUTUBE_SCOPE = ['https://www.googleapis.com/auth/youtube.readonly']
CONTENT_OWNER_ID = os.getenv('CONTENT_OWNER_ID')
DB_CONFIG = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME')
}
DEBUG = os.getenv('DEBUG')

def main():
    args = parse_arguments()
    youtube_data = get_authenticated_services()

    video_ids = args.id.split(',')
    result = []

    for i in range(0, len(video_ids), 50):
        video_ids_batch = video_ids[i:i + 50]

        if DEBUG:
            print(f"[DEBUG] video ids batch: {video_ids_batch}")

        result.extend(get_video_metadata(video_ids_batch, youtube_data))

    print(json.dumps(result, indent=2))

def get_authenticated_services():
    credentials = service_account.Credentials.from_service_account_file(YOUTUBE_SERVICE_ACCOUNT_FILE, scopes=YOUTUBE_SCOPE)
    youtube_data = build('youtube', 'v3', credentials=credentials)

    if DEBUG:
        print("[DEBUG] YouTube API auth")

    return youtube_data

def parse_arguments():
    parser = argparse.ArgumentParser(description='get video and video related channel metadata')
    parser.add_argument('--id', required=True, help='video id or ids to check (can be a comma separated list like: --id=g8jfnZzPLEI,UHpmgeWTyco,gablublu)')
    return parser.parse_args()

def db_connection():
    connection = mysql.connector.connect(**DB_CONFIG)
    return connection


def get_video_upload_frequency(video_count, channel_creation_date):
    if channel_creation_date is None or video_count == 0: return 0

    creation_date = datetime.fromisoformat(channel_creation_date.replace('Z', '+00:00'))
    days_since_creation = (datetime.now(timezone.utc) - creation_date).days

    if days_since_creation == 0:
        days_since_creation = 1

    videos_per_day = video_count / days_since_creation
    videos_per_week = videos_per_day * 7
    videos_per_month = videos_per_day * 30.44  # average days per month
    videos_per_year = videos_per_day * 365.25

    return {
        'videos_per_day': round(videos_per_day, 2),
        'videos_per_week': round(videos_per_week, 2),
        'videos_per_month': round(videos_per_month, 2),
        'videos_per_year': round(videos_per_year, 2),
        'total_videos': video_count,
        'days_active': days_since_creation
    }

def get_video_metadata(video_ids, youtube_data):
    videos = [{"video_id": vid, "metadata": {}} for vid in video_ids]
    videos_dict = {v["video_id"]: v for v in videos}

    try:
        request = youtube_data.videos().list(
            part='snippet,statistics,status,contentDetails,topicDetails',
            id=','.join(video_ids)
        )
        response = request.execute()

        channel_ids = list(set(
            video.get('snippet', {}).get('channelId')
            for video in response['items']
            if video.get('snippet', {}).get('channelId')
        ))

        channel_data = get_channel_metadata(channel_ids, youtube_data)

        for video in response['items']:
            channel_id = video.get('snippet', {}).get('channelId')
            channel_info = channel_data.get(channel_id, {})
            upload_frequency = channel_info.get('uploadFrequency')

            metadata: dict[str, dict[str, Any] | str | dict[str, str | Any] | Any] = {
                "video_id": video['id'],
                "video_title": video.get('snippet', {}).get('title'),
                "video_description": video.get('snippet', {}).get('description'),
                "video_tags": video.get('snippet', {}).get('tags', []),
                "video_category": video.get('snippet', {}).get('categoryId'),
                "length": video.get('contentDetails', {}).get('duration'),
                "published_date": video.get('snippet', {}).get('publishedAt'),
                "claimed_date": "",
                "engagement": {
                    "view_count": video.get('statistics', {}).get('viewCount'),
                    "like_count": video.get('statistics', {}).get('likeCount'),
                    "dislike_count": video.get('statistics', {}).get('dislikeCount'),
                    "comment_count": video.get('statistics', {}).get('commentCount'),
                    "favorite_count": video.get('statistics', {}).get('favoriteCount')
                },
                "channel": {
                    "id": channel_id,
                    "title": video.get('snippet', {}).get('channelTitle'),
                    "subscriber_count": channel_info.get('statistics', {}).get('subscriberCount', 0),
                    "verification_status": channel_info.get('status', {}).get('privacyStatus', ''),
                    "topic": ','.join(channel_info.get('topicDetails', {}).get('topicCategories', [])),
                    "creation_date": channel_info.get('snippet', {}).get('publishedAt', ''),
                    "video_upload_frequency": get_video_upload_frequency(int(channel_info.get('statistics', {}).get('videoCount', 0)), channel_info.get('snippet', {}).get('publishedAt', None))
                }
            }

            videos_dict[video['id']]["metadata"] = metadata

    except HttpError as e:
        print(f"[ERROR] {e}")

    return list(videos_dict.values())


def get_channel_metadata(channel_ids, youtube_data):
    channel_data = {}

    if channel_ids:
        channelResponse = youtube_data.channels().list(
            part='snippet,statistics,status,contentDetails,topicDetails',
            id=','.join(channel_ids)
        ).execute()

        channel_data = {
            channel['id']: channel
            for channel in channelResponse.get('items', [])
        }

    return channel_data

if __name__ == '__main__':
    main()
