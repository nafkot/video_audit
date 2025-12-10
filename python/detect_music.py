import os
import sys
import json
from acrcloud.recognizer import ACRCloudRecognizer
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Detect music in an audio file")
    parser.add_argument("--id", required=True, help="Filename of the audio track to analyze")
    return parser.parse_args()

def structure_result(raw_json):
    """Normalize and prettify the ACRCloud JSON response."""
    try:
        data = json.loads(raw_json)
    except Exception:
        return {"error": "Invalid JSON returned from ACRCloud", "raw": raw_json}

    if "metadata" not in data or "music" not in data["metadata"]:
        return {"status": data.get("status"), "message": "No music recognized"}

    structured_tracks = []

    for track in data["metadata"]["music"]:
        structured_tracks.append({
            "acrid": track.get("acrid"),
            "title": track.get("title"),
            "artists": [a.get("name") for a in track.get("artists", [])],
            "album": track.get("album", {}).get("name"),
            "score": track.get("score"),
            "label": track.get("label"),
            "genres": [g.get("name") for g in track.get("genres", [])] if track.get("genres") else None,
            "duration_ms": track.get("duration_ms"),
            "release_date": track.get("release_date"),

            "timing": {
                "sample_begin": track.get("sample_begin_time_offset_ms"),
                "sample_end": track.get("sample_end_time_offset_ms"),
                "db_begin": track.get("db_begin_time_offset_ms"),
                "db_end": track.get("db_end_time_offset_ms"),
                "play_offset": track.get("play_offset_ms")
            },

            "external_ids": track.get("external_ids", {}),

            "external_metadata": track.get("external_metadata", {})
        })

    return {
        "status": data.get("status", {}),
        "timestamp_utc": data.get("metadata", {}).get("timestamp_utc"),
        "tracks": structured_tracks
    }


def analyze_track():
    args = parse_arguments()

    AUDIO_BASE_PATH = "/home/merhawi/analyser/sync-music-command/storage/audio"
    desktop_path = os.path.join(AUDIO_BASE_PATH, args.id)

    if not os.path.exists(desktop_path):
        print(json.dumps({"error": f"File not found: {desktop_path}"}))
        sys.exit(1)

    config = {
        'host': 'identify-eu-west-1.acrcloud.com',
        'access_key': '3733b3898892aee4d46d2c56cc2df140',
        'access_secret': 'fo8mHNBNW66qJ3Bc3A7VsotR6U4cQCIo1XpSwqLc',
        'recognize_type': 2,
    }

    #desktop_path = "/home/merhawi/analyser/sync-music-command/storage/audio/0kfLmie73dA_instrumental.mp3"

    if not os.path.exists(desktop_path):
        print(json.dumps({"error": f"File does not exist: {desktop_path}"}, indent=4))
        return

    recognizer = ACRCloudRecognizer(config)

    print(f"Analyzing file")
    print("-" * 50)

    try:
        raw_result = recognizer.recognize_by_file(desktop_path, 0, 10)

        structured = structure_result(raw_result)

        print(json.dumps(structured, indent=4))

    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=4))
        sys.exit(1)


if __name__ == '__main__':
    analyze_track()

