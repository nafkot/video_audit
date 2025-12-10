import os
import re
import csv
import time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- CONFIGURATION ---
SERVICE_ACCOUNT_FILE = 'account.json'
OUTPUT_FILE = 'cms_leads_api.csv'

# Target "Mid-Tier" niches to avoid major labels (Sony/Universal)
# We want keywords likely to have indie/mid-market distributors.
SEARCH_KEYWORDS = [
    "Afrobeats 2025 official video", "Techno mix 2025", "Deep House 2025",
    "Trap beats instrumental", "Lo-fi hip hop original", "Reggaeton 2025",
    "Underground Rap", "Piano classical music", "Meditation music original",
    "Vlog music no copyright", "Gaming highlights 2025", "Psytrance 2025",
    "Indie Rock 2025", "Amapiano 2025", "Balkan Pop 2025",
    "K-Pop indie", "J-Pop 2025", "Russian Rap 2025", "French Hip Hop 2025",
    "Brazilian Phonk", "Synthwave 2025", "Dungeon Synth",
    "Gym Phonk", "Speedcore", "Nightcore 2025"
]

SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']

def get_authenticated_service():
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        return build('youtube', 'v3', credentials=creds)
    except Exception as e:
        print(f"Error authenticating: {e}")
        return None

def extract_provider(description):
    """
    Finds 'Provided to YouTube by [Company]' in the full description.
    """
    if not description:
        return None
    # Regex to capture the company name after the standard phrase
    match = re.search(r"Provided to YouTube by (.+)", description)
    if match:
        return match.group(1).strip()
    return None

def main():
    service = get_authenticated_service()
    if not service:
        return

    # Load existing companies to avoid duplicates
    existing_companies = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    existing_companies.add(row[0])

    # Open CSV in Append mode
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header only if file is new
        if os.stat(OUTPUT_FILE).st_size == 0:
            writer.writerow(["Company Name", "Keyword", "Video ID"])

        print(f"[*] Starting API Scan using {SERVICE_ACCOUNT_FILE}...")
        print(f"[*] Daily Quota Limit: ~90 Searches. running {len(SEARCH_KEYWORDS)} keywords.")

        for keyword in SEARCH_KEYWORDS:
            print(f"\n--- Searching: {keyword} ---")
            
            try:
                # 1. SEARCH (Cost: 100 Units)
                search_response = service.search().list(
                    q=keyword,
                    part="id",
                    maxResults=50,  # Max allowed per page
                    type="video"
                ).execute()

                video_ids = []
                for item in search_response.get('items', []):
                    if 'id' in item and 'videoId' in item['id']:
                        video_ids.append(item['id']['videoId'])
                
                if not video_ids:
                    print("No videos found.")
                    continue

                # 2. GET DETAILS (Cost: 1 Unit)
                # We need this because search().list truncates descriptions.
                # videos().list gives the FULL description where the "Provided by" line lives.
                videos_response = service.videos().list(
                    id=','.join(video_ids),
                    part="snippet"
                ).execute()

                for video in videos_response.get('items', []):
                    description = video['snippet'].get('description', '')
                    provider = extract_provider(description)

                    if provider:
                        # Filter out the "Majors" to keep the list useful
                        if provider not in existing_companies and "Sony" not in provider and "Universal" not in provider:
                            print(f"[+] FOUND: {provider}")
                            writer.writerow([provider, keyword, video['id']])
                            existing_companies.add(provider)
                            f.flush() # Save immediately

            except HttpError as e:
                if e.resp.status in [403, 429]:
                    print("FATAL: Quota Exceeded for today. Script stopping.")
                    break
                print(f"API Error: {e}")

    print(f"\n[DONE] Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
