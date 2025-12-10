#!/usr/bin/env python3

import os
import json
import argparse
import pathlib
import subprocess
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import requests


def ensure_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: pathlib.Path, session: requests.Session, desc: str = ""):
    ensure_dir(dest.parent)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SyncVaultBot/1.0; +https://syncvault.com)",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    print(f"  ↓ Downloading: {url.split('?')[0]}... -> {dest}")
    resp = session.get(url, headers=headers, stream=True, timeout=60)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"    ✖ Failed to download {desc or dest.name}: {e}")
        return False

    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return True


def save_metadata(data: dict, dest_dir: pathlib.Path):
    ensure_dir(dest_dir)
    out_file = dest_dir / "yt_api_metadata.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✔ Saved metadata to {out_file}")


def download_thumbnails(data: dict, dest_dir: pathlib.Path, session: requests.Session):
    thumbs = data.get("thumbnail") or []
    if not thumbs:
        return
    tdir = dest_dir / "thumbnails"
    ensure_dir(tdir)
    for thumb in thumbs:
        url = thumb.get("url")
        if not url:
            continue
        w = thumb.get("width")
        h = thumb.get("height")
        suffix = pathlib.Path(urlparse(url).path).suffix or ".jpg"
        name = f"thumb_{w}x{h}{suffix}"
        download_file(url, tdir / name, session, desc=name)


def download_storyboards(data: dict, dest_dir: pathlib.Path, session: requests.Session):
    sbs = data.get("storyboards") or []
    if not sbs:
        return
    sbdir = dest_dir / "storyboards"
    ensure_dir(sbdir)
    for i, sb in enumerate(sbs):
        urls = sb.get("url") or []
        for j, url in enumerate(urls):
            suffix = pathlib.Path(urlparse(url).path).suffix or ".jpg"
            name = f"storyboard_{i}_part{j}{suffix}"
            download_file(url, sbdir / name, session, desc=name)


def build_json3_caption_url(base_url: str) -> str:
    """
    Take the caption baseUrl from yt-api (timedtext) and force fmt=json3
    so we can get machine-readable captions.
    """
    parsed = urlparse(base_url)
    qs = parse_qs(parsed.query)
    qs["fmt"] = ["json3"]
    new_query = urlencode(qs, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def download_captions(data: dict, dest_dir: pathlib.Path, session: requests.Session):
    caps = (data.get("captions") or {}).get("captionTracks") or []
    if not caps:
        print("  ℹ No captionTracks found.")
        return
    cdir = dest_dir / "captions"
    ensure_dir(cdir)
    for track in caps:
        base_url = track.get("baseUrl")
        lang = track.get("languageCode") or track.get("vssId", "").lstrip("a.")
        name = track.get("name") or lang
        if not base_url:
            continue

        json3_url = build_json3_caption_url(base_url)
        print(f"  ↓ Fetching captions for [{lang}] {name} ...")
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; SyncVaultBot/1.0; +https://syncvault.com)",
            "Accept": "*/*",
        }
        resp = session.get(json3_url, headers=headers, timeout=60)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            print(f"    ✖ Failed to fetch captions for {lang}: {e}")
            continue

        try:
            raw = resp.json()
        except json.JSONDecodeError:
            # Save raw text if not JSON (e.g., xml)
            raw = {"raw": resp.text}

        raw_path = cdir / f"captions_{lang}_raw.json"
        with raw_path.open("w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)

        # Build clean segments if json3 structure
        segments = []
        events = raw.get("events", []) if isinstance(raw, dict) else []
        for ev in events:
            t_start = ev.get("tStartMs")
            dur = ev.get("dDurationMs")
            segs = ev.get("segs") or []
            text = "".join(s.get("utf8", "") for s in segs if "utf8" in s)
            if not text.strip():
                continue
            start_sec = float(t_start) / 1000.0 if t_start is not None else None
            dur_sec = float(dur) / 1000.0 if dur is not None else None
            segments.append(
                {
                    "start": start_sec,
                    "duration": dur_sec,
                    "text": text.strip(),
                }
            )

        clean_path = cdir / f"captions_{lang}_clean.json"
        with clean_path.open("w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)

        print(f"    ✔ Saved captions for {lang} -> {clean_path}")


def pick_best_mp4(formats_list):
    """Strategy A: pick best video/mp4 with audio from 'formats' list."""
    if not formats_list:
        return None
    candidates = [
        f for f in formats_list
        if isinstance(f, dict)
        and isinstance(f.get("mimeType"), str)
        and f["mimeType"].startswith("video/mp4")
    ]
    if not candidates:
        return None
    # Sort by height then bitrate
    candidates.sort(key=lambda f: (int(f.get("height", 0)), int(f.get("bitrate", 0))), reverse=True)
    return candidates[0]


def pick_itag(itag, formats_list, adaptive_list):
    for coll in (formats_list or []), (adaptive_list or []):
        for f in coll:
            if int(f.get("itag", -1)) == itag:
                return f
    return None


def pick_best_video_audio(adaptive_list):
    """Strategy C: bestvideo + bestaudio from adaptiveFormats."""
    if not adaptive_list:
        return None, None
    videos = [
        f for f in adaptive_list
        if isinstance(f.get("mimeType"), str)
        and f["mimeType"].startswith("video/")
    ]
    audios = [
        f for f in adaptive_list
        if isinstance(f.get("mimeType"), str)
        and f["mimeType"].startswith("audio/")
    ]
    if not videos or not audios:
        return None, None
    videos.sort(key=lambda f: (int(f.get("height", 0)), int(f.get("bitrate", 0))), reverse=True)
    audios.sort(key=lambda f: int(f.get("bitrate", 0)), reverse=True)
    return videos[0], audios[0]


def extract_audio_from_video(video_path: pathlib.Path, audio_path: pathlib.Path):
    """
    Convert a downloaded video into mono 16kHz WAV suitable for Whisper.
    Requires ffmpeg to be installed.
    """
    ensure_dir(audio_path.parent)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac", "1",         # mono
        "-ar", "16000",     # 16 kHz
        "-vn",              # no video
        str(audio_path),
    ]
    print(f"  🎧 Extracting mono audio with ffmpeg -> {audio_path}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("    ✔ Audio extraction complete.")
    except FileNotFoundError:
        print("    ✖ ffmpeg not found. Please install ffmpeg to enable audio extraction.")
    except subprocess.CalledProcessError as e:
        print(f"    ✖ ffmpeg failed: {e}")


def handle_video_and_audio(data: dict, dest_dir: pathlib.Path, session: requests.Session,
                           itag: int | None, bestvideo_bestaudio: bool, no_audio: bool):
    """
    Apply format strategy:
      A (default): best MP4 from 'formats'
      B (--itag): specific itag from formats/adaptiveFormats
      C (--bestvideo+bestaudio): merge streams (download both)
    Then (unless --no-audio), extract mono WAV audio from the chosen video.
    """
    vid = data.get("id")
    formats_list = data.get("formats") or []
    adaptive_list = data.get("adaptiveFormats") or []

    vdir = dest_dir / "videos"
    ensure_dir(vdir)

    chosen_video_file = None

    # Strategy B: explicit itag
    if itag is not None:
        fmt = pick_itag(itag, formats_list, adaptive_list)
        if not fmt:
            print(f"  ✖ Could not find format with itag={itag}")
        else:
            url = fmt.get("url")
            if url:
                height = fmt.get("height")
                ext = ".mp4"
                fname = f"{itag}_{height or 'na'}{ext}"
                video_path = vdir / fname
                if download_file(url, video_path, session, desc=f"itag {itag}"):
                    chosen_video_file = video_path

    # Strategy C: bestvideo+bestaudio
    elif bestvideo_bestaudio:
        v_fmt, a_fmt = pick_best_video_audio(adaptive_list)
        if not v_fmt or not a_fmt:
            print("  ✖ Could not find suitable bestvideo/bestaudio streams.")
        else:
            v_url = v_fmt.get("url")
            a_url = a_fmt.get("url")
            if v_url and a_url:
                v_ext = ".mp4"
                a_ext = ".m4a"
                v_path = vdir / f"bestvideo_{v_fmt.get('height','na')}{v_ext}"
                a_path = vdir / f"bestaudio{a_ext}"
                ok_v = download_file(v_url, v_path, session, desc="bestvideo")
                ok_a = download_file(a_url, a_path, session, desc="bestaudio")
                if ok_v and ok_a:
                    merged_path = vdir / f"{vid}_merged.mp4"
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(v_path),
                        "-i", str(a_path),
                        "-c", "copy",
                        str(merged_path),
                    ]
                    print(f"  🎬 Merging bestvideo+bestaudio -> {merged_path}")
                    try:
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        print("    ✔ Merge complete.")
                        chosen_video_file = merged_path
                    except FileNotFoundError:
                        print("    ✖ ffmpeg not found. Skipping merge.")
                    except subprocess.CalledProcessError as e:
                        print(f"    ✖ ffmpeg merge failed: {e}")

    # Strategy A: best MP4 (default)
    else:
        fmt = pick_best_mp4(formats_list)
        if not fmt:
            print("  ✖ No suitable video/mp4 format found in 'formats'.")
        else:
            url = fmt.get("url")
            if url:
                height = fmt.get("height")
                itag_val = fmt.get("itag")
                ext = ".mp4"
                fname = f"{itag_val}_{height or 'na'}{ext}"
                video_path = vdir / fname
                if download_file(url, video_path, session, desc="best mp4"):
                    chosen_video_file = video_path

    # Audio extraction if requested
    if not no_audio and chosen_video_file is not None:
        audio_dir = dest_dir / "audio"
        audio_path = audio_dir / "audio_mono_16k.wav"
        extract_audio_from_video(chosen_video_file, audio_path)
    elif no_audio:
        print("  ℹ --no-audio flag set; skipping audio extraction.")


def process_yt_api_json(input_path: str, output_dir: str,
                        itag: int | None, bestvideo_bestaudio: bool, no_audio: bool):
    input_file = pathlib.Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input JSON file not found: {input_path}")

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Support either single-object or list-of-objects JSON
    videos = data if isinstance(data, list) else [data]

    session = requests.Session()

    for vid_data in videos:
        video_id = vid_data.get("id")
        channel_id = vid_data.get("channelId") or "unknown_channel"
        print(f"Processing video {video_id} from channel {channel_id}")

        base_dir = pathlib.Path(output_dir) / channel_id / video_id
        ensure_dir(base_dir)

        save_metadata(vid_data, base_dir)
        download_thumbnails(vid_data, base_dir, session)
        download_storyboards(vid_data, base_dir, session)
        download_captions(vid_data, base_dir, session)
        handle_video_and_audio(
            vid_data,
            base_dir,
            session,
            itag=itag,
            bestvideo_bestaudio=bestvideo_bestaudio,
            no_audio=no_audio,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download YouTube assets (thumbnails, storyboards, captions, optional audio) "
                    "from a yt-api JSON response."
    )
    parser.add_argument(
        "--input",
        default="YT-API_result.json",
        help="Path to the yt-api JSON response file (default: YT-API_result.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="output_assets",
        help="Base output directory (default: output_assets)",
    )
    # Format selection
    parser.add_argument(
        "--itag",
        type=int,
        help="Download specific itag (strategy B). If omitted, use best MP4 or bestvideo+bestaudio.",
    )
    parser.add_argument(
        "--bestvideo+bestaudio",
        dest="bestvideo_bestaudio",
        action="store_true",
        help="Use bestvideo+bestaudio strategy (strategy C) instead of single MP4.",
    )
    # Audio control
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Do NOT extract mono audio (only download metadata/thumbnails/storyboards/captions).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    process_yt_api_json(
        input_path=args.input,
        output_dir=args.output_dir,
        itag=args.itag,
        bestvideo_bestaudio=args.bestvideo_bestaudio,
        no_audio=args.no_audio,
    )


if __name__ == "__main__":
    main()

