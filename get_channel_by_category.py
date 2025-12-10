#!/usr/bin/env python3

import os
import argparse
import csv
import time
from datetime import datetime, timezone, timedelta

from google.oauth2 import service_account
from googleapiclient.discovery import build

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SERVICE_ACCOUNT_FILE = "account.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]

# Beauty Topic IDs
BEAUTY_TOPIC_IDS = {
    "/m/01jdpf",   # Makeup
    "/m/01d34b",   # Beauty
    "/m/0463cq",   # Cosmetics
    "/m/02jjt",    # Hairstyle
    "/m/019_rr",   # Fashion
    "/m/02fq_2",   # Skin care
    "/m/02fq_7",   # Nail care
    "/m/02wbm"     # Lifestyle
}

DEFAULT_KEYWORDS = [
    "makeup", "beauty", "cosmetics", "skincare",
    "makeup tutorial", "nail art", "hair tutorial",
    "beauty tips", "mua"
]

# Cache the YouTube client to reduce quota
_youtube = None

def youtube_client():
    global _youtube
    if _youtube is None:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        _youtube = build("youtube", "v3", credentials=credentials, static_discovery=False)
    return _youtube


# ---------------------------------------------------------
# SEARCH CHANNELS
# ---------------------------------------------------------
def search_channels(keyword, limit):
    youtube = youtube_client()

    req = youtube.search().list(
        part="snippet",
        q=keyword,
        type="channel",
        maxResults=limit,
        order="viewCount"
    )
    resp = req.execute()

    return [item["snippet"]["channelId"] for item in resp.get("items", [])]


# ---------------------------------------------------------
# GET CHANNEL METADATA
# ---------------------------------------------------------
def get_channel_info(channel_ids):
    youtube = youtube_client()

    req = youtube.channels().list(
        part="snippet,statistics,topicDetails,contentDetails",
        id=",".join(channel_ids)
    )
    resp = req.execute()

    channels = []
    for ch in resp.get("items", []):
        channels.append({
            "channel_id": ch["id"],
            "title": ch["snippet"]["title"],
            "description": ch["snippet"].get("description", ""),
            "country": ch["snippet"].get("country", ""),
            "subscribers": int(ch["statistics"].get("subscriberCount", 0)),
            "views": int(ch["statistics"].get("viewCount", 0)),
            "video_count": int(ch["statistics"].get("videoCount", 0)),
            "topics": ch.get("topicDetails", {}).get("topicIds", []),
            "uploads_playlist": ch.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads"),
            "created_at": ch["snippet"]["publishedAt"]
        })

    return channels


# ---------------------------------------------------------
# BEAUTY FILTER
# ---------------------------------------------------------
def is_beauty(topics):
    return any(t in BEAUTY_TOPIC_IDS for t in topics)


# ---------------------------------------------------------
# GET RECENT UPLOADS + AVERAGE VIEWS LOGIC
# ---------------------------------------------------------
def get_upload_activity(playlist_id):
    if not playlist_id:
        return {}

    youtube = youtube_client()
    now = datetime.now(timezone.utc)

    cutoff_7 = now - timedelta(days=7)
    cutoff_30 = now - timedelta(days=30)
    cutoff_90 = now - timedelta(days=90)

    last_upload_date = None
    videos_7 = videos_30 = videos_90 = 0
    views_last_30 = []

    next_page = None

    while True:
        req = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page
        )
        resp = req.execute()

        break_outer = False

        ids_to_fetch = []
        video_times = {}

        for it in resp.get("items", []):
            published_at = it["snippet"]["publishedAt"]
            dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            vid = it["contentDetails"]["videoId"]

            video_times[vid] = dt
            ids_to_fetch.append(vid)

            if last_upload_date is None or dt > last_upload_date:
                last_upload_date = dt

            if dt >= cutoff_7:
                videos_7 += 1
            if dt >= cutoff_30:
                videos_30 += 1
            if dt >= cutoff_90:
                videos_90 += 1
            else:
                break_outer = True
                break

        # Fetch statistics for these videos
        if ids_to_fetch:
            stat_req = youtube.videos().list(
                part="statistics",
                id=",".join(ids_to_fetch)
            )
            stat_resp = stat_req.execute()

            for v in stat_resp.get("items", []):
                vid = v["id"]
                vcount = int(v.get("statistics", {}).get("viewCount", 0))
                dt = video_times[vid]

                if dt >= cutoff_30:
                    views_last_30.append(vcount)

        if break_outer:
            break

        if "nextPageToken" in resp:
            next_page = resp["nextPageToken"]
            time.sleep(0.3)
        else:
            break

    days_since_last = (now - last_upload_date).days if last_upload_date else None
    avg_views_last_30 = round(sum(views_last_30) / len(views_last_30), 2) if views_last_30 else 0

    return {
        "videos_last_7_days": videos_7,
        "videos_last_30_days": videos_30,
        "videos_last_90_days": videos_90,
        "last_upload_date": last_upload_date.isoformat() if last_upload_date else "",
        "days_since_last_upload": days_since_last,
        "avg_uploads_per_week": round(videos_30 / 4, 2),
        "avg_uploads_per_month": videos_30,
        "avg_views_last_30_days": avg_views_last_30
    }


# ---------------------------------------------------------
# WRITE CSV
# ---------------------------------------------------------
def save_csv(rows):
    with open("beauty_channels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Rank", "Channel ID", "Title", "Subscribers", "Views",
            "Country", "Total Videos",
            "Videos Last 7 Days", "Videos Last 30 Days", "Videos Last 90 Days",
            "Avg Uploads/Week", "Avg Uploads/Month",
            "Avg Views Last 30 Days",
            "Days Since Last Upload", "Last Upload Date",
            "Topics", "Description"
        ])
        for i, ch in enumerate(rows, start=1):
            w.writerow([
                i, ch["channel_id"], ch["title"], ch["subscribers"], ch["views"],
                ch["country"], ch["video_count"],
                ch["videos_last_7_days"], ch["videos_last_30_days"], ch["videos_last_90_days"],
                ch["avg_uploads_per_week"], ch["avg_uploads_per_month"],
                ch["avg_views_last_30_days"],
                ch["days_since_last_upload"], ch["last_upload_date"],
                ",".join(ch["topics"]), ch["description"]
            ])


# ---------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------
def parse_arguments():
    p = argparse.ArgumentParser()

    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--keywords")
    p.add_argument("--min-subscriber", type=int, default=1_000_000)
    p.add_argument("--country")
    p.add_argument("--active-only", action="store_true")
    p.add_argument("--min-30day-uploads", type=int, default=0)
    p.add_argument("--sort", choices=["subscribers", "activity"], default="subscribers")

    return p.parse_args()


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    args = parse_arguments()

    limit = args.limit
    keywords = args.keywords.split(",") if args.keywords else DEFAULT_KEYWORDS
    min_subs = args.min_subscriber
    allowed_countries = args.country.split(",") if args.country else None

    unique = set()

    print("\n🔎 Searching for channels...")
    for kw in keywords:
        ids = search_channels(kw, limit)
        unique.update(ids)
        print(f"   → {kw}: {len(ids)} results")

    ids_list = list(unique)
    beauty_channels = []

    print("\n📡 Fetching channel metadata...")
    for i in range(0, len(ids_list), 50):
        batch = ids_list[i:i+50]
        info = get_channel_info(batch)

        for ch in info:
            if ch["subscribers"] < min_subs:
                continue
            if allowed_countries and ch["country"] not in allowed_countries:
                continue
            if not is_beauty(ch["topics"]):
                continue

            uploads = get_upload_activity(ch["uploads_playlist"])
            ch.update(uploads)

            if args.active_only and ch["videos_last_30_days"] == 0:
                continue
            if ch["videos_last_30_days"] < args.min_30day_uploads:
                continue

            beauty_channels.append(ch)

        time.sleep(0.3)

    # Sort
    if args.sort == "activity":
        beauty_channels.sort(key=lambda x: (x["videos_last_30_days"], x["avg_views_last_30_days"]), reverse=True)
    else:
        beauty_channels.sort(key=lambda x: x["subscribers"], reverse=True)

    save_csv(beauty_channels)

    print("\n💾 Saved to beauty_channels.csv")
    print(f"🎀 Total beauty channels: {len(beauty_channels)}")


if __name__ == "__main__":
    main()

