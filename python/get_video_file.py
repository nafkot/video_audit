#!/usr/bin/env python3

import os
import re
import argparse
import json
import glob
import dotenv
import yt_dlp
import requests
import random
from urllib.parse import urlencode, urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed

class SilentLogger:
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

dotenv.load_dotenv()

DEBUG = os.getenv('DEBUG')
CONTENT_PATH = os.getenv('CONTENT_PATH')
SCRAPERAPI_KEY = os.getenv('SCRAPERAPI_KEY')

PROXY_LIST_URL = "https://free.redscrape.com/api/proxies"
IP_TEST_URL = "http://api.ipify.org"
WORKING_PROXIES = []

def main():
    args = parse_arguments()
    os.makedirs(f"{CONTENT_PATH}/videos", exist_ok=True)

    video_ids = []
    results = []
    service = args.service

    if args.id:
        video_ids = args.id.split(',')

    if args.url:

        urls = args.url.split(',')

        for url in urls:
            parsed_url = parse_video_url(url)

            if parsed_url['service'] == 'tiktok':
                video_ids.append(parsed_url['user']+'|'+parsed_url['video_id'])
            else:
                video_ids.append(parsed_url['video_id'])

    with ThreadPoolExecutor(max_workers=(os.cpu_count() * 10)) as executor:
        futures = [executor.submit(download_video, video_id, 0, False, service) for video_id in video_ids]
        for future in as_completed(futures):
            results.append(future.result())

    print(json.dumps(results, indent=2))

def parse_arguments():
    parser = argparse.ArgumentParser(description='get video file')
    parser.add_argument('--id', help='video id or ids to download')
    parser.add_argument('--url', help='video url or urls to download')
    parser.add_argument('--no-proxy', action='store_true', help='disable proxy usage')
    parser.add_argument('--service', default='youtube')
    args = parser.parse_args()

    if not args.id and not args.url:
        parser.error('--url OR --id is required')

    return args

def is_valid_youtube_id(video_id: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9_-]{11}$', video_id))

def get_real_ip():
    try:
        return requests.get(IP_TEST_URL, timeout=5).text.strip()
    except Exception as e:
        # print(e)
        return None

def test_proxy(proxy, real_ip):
    try:
        proxy_base_ip = proxy.split('//')[1].split(':')[0] if '//' in proxy else proxy.split(':')[0]
        response = requests.get(IP_TEST_URL, proxies={'http': proxy_base_ip, 'https': proxy_base_ip}, timeout=5)

        if response.status_code != 200:
            return False

        return proxy_base_ip != real_ip
    except Exception as e:
        print(e)
        return False

def fetch_proxies():
    global WORKING_PROXIES
    real_ip = get_real_ip()

    if not real_ip:
        WORKING_PROXIES = []
        return

    try:
        response = requests.get(PROXY_LIST_URL, timeout=10)
        proxy_list = [p.strip() for p in response.text.strip().split('\n') if p.strip()][:50]

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(lambda p: (p, test_proxy(p, real_ip)), proxy_list))
            WORKING_PROXIES = [p for p, works in results if works]

    except Exception as e:
        WORKING_PROXIES = []

def get_random_proxy():
    return random.choice(WORKING_PROXIES) if WORKING_PROXIES else None


def parse_video_url(url):
    # YouTube
    if 'youtube.com' in url or 'youtu.be' in url:
        # youtu.be/VIDEO_ID
        if 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[-1].split('?')[0]
            return {'service': 'youtube', 'video_id': video_id, 'user': None}

        # youtube.com/watch?v=VIDEO_ID or youtube.com/shorts/VIDEO_ID
        parsed = urlparse(url)
        if 'watch' in url:
            video_id = parse_qs(parsed.query).get('v', [None])[0]
        elif 'shorts/' in url:
            video_id = url.split('shorts/')[-1].split('?')[0]
        else:
            return None

        if video_id:
            return {'service': 'youtube', 'video_id': video_id, 'user': None}

    # Instagram
    elif 'instagram.com' in url:
        match = re.search(r'/(p|reel)/([A-Za-z0-9_-]+)', url)
        if match:
            video_id = match.group(2)
            return {'service': 'instagram', 'video_id': video_id, 'user': None}

    # TikTok
    elif 'tiktok.com' in url:
        match = re.search(r'@([^/]+)/video/(\d+)', url)
        if match:
            user = match.group(1)
            video_id = match.group(2)
            return {'service': 'tiktok', 'video_id': video_id, 'user': user}

        if 'vm.tiktok.com' in url or 'vt.tiktok.com' in url:
            match = re.search(r'/(vm|vt)\.tiktok\.com/([A-Za-z0-9]+)', url)
            if match:
                video_id = match.group(2)
                return {'service': 'tiktok', 'video_id': video_id, 'user': None}

    return None

def download_video(video_id, attempt=0, use_proxy=False, service = 'youtube'):
    print(service)
    if use_proxy and len(WORKING_PROXIES) == 0:
        fetch_proxies()

    use_rapidapi_proxy = True
    proxy = None

    output_path = f"{CONTENT_PATH}/videos/{video_id}.%(ext)s"

    ydl_opts = {
        'outtmpl': output_path,
        'format': 'best',
        'socket_timeout': 30,
        'retries': 2,
        'retry_sleep': 2,
        'fragment_retries': 5,
        'nocheckcertificate': True,
    }

    if service == 'youtube':
        if len(glob.glob(f"{CONTENT_PATH}/videos/{video_id}.*")) > 0 or not is_valid_youtube_id(video_id):
            return {'id': video_id, 'success': False, 'attempt': attempt}

        url = f"https://www.youtube.com/watch?v={video_id}"
    elif service == 'instagram':
        url = f"https://instagram.com/reel/{video_id}/"

        ydl_opts.update({
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            }
        })
    elif service == 'tiktok':
        ydl_opts.update({
            # 'impersonate': 'chrome-131-android-14',
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://www.tiktok.com/',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Ch-Ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"Windows"',
            },
            'format': 'best[ext=mp4]/best',  # Remove extractor_args - they may be causing issues
            'cookiefile': None,  # Don't use cookies
        })

        partials = video_id.split('|')#format is https://www.tiktok.com/@raph.desroches7/video/7571589872381693191
        url = f"https://www.tiktok.com/@{partials[0]}/video/{partials[1]}"

    if use_proxy:
        proxy = get_random_proxy()

        if proxy:
            ydl_opts['proxy'] = proxy
        else:
            return download_video(video_id, attempt + 1, True, service)

    elif service == 'instagram':
    # elif service == 'tiktok' or service == 'instagram':
        scraperapi_proxy = f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001"
        ydl_opts['proxy'] = scraperapi_proxy
    elif service == 'tiktok':
        if DEBUG:
            print('tiktok no proxy')#try without proxy first
    else:
        base = f"https://api.scraperapi.com/?api_key={SCRAPERAPI_KEY}"
        target = url
        url = f"{base}&{urlencode({'url': target})}"

    if not DEBUG:
        ydl_opts['quiet'] = True
        ydl_opts['no_warnings'] = True
        ydl_opts['logger'] = SilentLogger()
        ydl_opts['progress_hooks'] = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            return {'id': video_id, 'success': True, 'attempt': attempt}
        except:
            if attempt < 2:
                if attempt > 1: use_proxy = True
                return download_video(video_id, attempt + 1, use_proxy, service)#recursion
            return {'id': video_id, 'success': False, 'attempt': attempt}

if __name__ == '__main__':
    main()