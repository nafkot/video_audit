#!/bin/bash

# A master script to download, frame-split, and analyse a video with CLIP.
#
# Usage:
# ./analyse_video_360.sh https://orcid.org/ [--service youtube|tiktok]
# Example (YouTube):
# ./analyse_video_360.sh UHpmgeWTyco
# Example (TikTok):
# ./analyse_video_360.sh https://www.tiktok.com/@xellis.millerx/video/7572202702285982998 --service tiktok

set -e # Exit immediately if any command fails

# --- NEW ARGUMENT PARSING ---
SERVICE="youtube"
INPUT_ARG=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --service) SERVICE="$2"; shift ;;
        *) INPUT_ARG="$1" ;;
    esac
    shift
done

if [ -z "$INPUT_ARG" ]; then
  echo "Error: No URL or ID provided."
  echo "Usage: ./analyse_video_360.sh https://orcid.org/ [--service youtube|tiktok]"
  exit 1
fi
# --- END NEW ARGUMENT PARSING ---


# Load .env file if it exists to get CONTENT_PATH
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Set default CONTENT_PATH if not set in .env
CONTENT_PATH=${CONTENT_PATH:-storage}
FPS=0.5 # Frames per second to extract

# --- NEW: Determine DOWNLOAD_ID and BASE_ID ---
if [ "$SERVICE" == "tiktok" ]; then
  echo "TikTok service selected. Parsing URL..."
  # Extract username and video ID from URL like: https://www.tiktok.com/@username/video/12345
  if [[ $INPUT_ARG =~ \/@([^\/]+)\/video\/([0-9]+) ]]; then
    USERNAME=${BASH_REMATCH[1]}
    BASE_ID=${BASH_REMATCH[2]}
    DOWNLOAD_ID="$USERNAME|$BASE_ID"
    echo "Parsed username: $USERNAME"
    echo "Parsed video ID (Base ID): $BASE_ID"
    echo "Download ID: $DOWNLOAD_ID"
  else
    echo "Error: Invalid TikTok URL format. Expected: https://.../@username/video/videoid"
    exit 1
  fi
elif [ "$SERVICE" == "youtube" ]; then
  echo "YouTube service selected."
  BASE_ID=$INPUT_ARG
  DOWNLOAD_ID=$INPUT_ARG
else
  echo "Error: Unknown service '$SERVICE'. Supported: youtube, tiktok"
  exit 1
fi
# --- END NEW ---


echo "--- [Step 1/4] Starting 360 Analysis for: $BASE_ID (Service: $SERVICE) ---"
echo "Using CONTENT_PATH: $CONTENT_PATH"

# --- 2. Check for existing video / Download ---
# Use $DOWNLOAD_ID for finding/downloading the file
VIDEO_FILE_PATH=$(find $CONTENT_PATH/videos -type f -name "$DOWNLOAD_ID.*" | head -n 1)

if [ -z "$VIDEO_FILE_PATH" ]; then
  echo "[Step 2/4] Video not found. Downloading..."
  # Pass --service and --id (which is DOWNLOAD_ID)
  python python/get_video_file.py --id "$DOWNLOAD_ID" --service "$SERVICE"
  
  VIDEO_FILE_PATH=$(find $CONTENT_PATH/videos -type f -name "$DOWNLOAD_ID.*" | head -n 1)
  
  if [ -z "$VIDEO_FILE_PATH" ]; then
    echo "Error: Failed to find downloaded video file for $DOWNLOAD_ID in $CONTENT_PATH/videos/"
    exit 1
  fi
  echo "Video saved to: $VIDEO_FILE_PATH"
else
  echo "[Step 2/4] Video already exists, skipping download."
  echo "Using existing file: $VIDEO_FILE_PATH"
fi

# --- 3. Split into Frames (if necessary) ---
VIDEO_FILENAME=$(basename "$VIDEO_FILE_PATH")
# Use $BASE_ID for the frame directory name
FRAME_DIR="$CONTENT_PATH/videos/frames/$BASE_ID"

if [ ! -d "$FRAME_DIR" ] || [ -z "$(ls -A "$FRAME_DIR")" ]; then
  echo "[Step 3/4] Frames not found. Splitting '$VIDEO_FILENAME' into frames (at $FPS fps)..."
  # Pass the new --output_name argument to use the clean BASE_ID for the folder
  python3 python/split_video_into_frames.py --fps $FPS --files "$VIDEO_FILENAME" --output_name "$BASE_ID"
  echo "Frames saved to: $FRAME_DIR"
else
  echo "[Step 3/4] Frames already exist, skipping split."
  echo "Using existing frames from: $FRAME_DIR"
fi

# --- 4. Run CLIP Analysis ---
echo "[Step 4/4] Analysing frames with CLIP service..."
# Use $FRAME_DIR (which is based on $BASE_ID)
python3 python/analyse_frames.py --frames_dir "$FRAME_DIR"

echo "--- Analysis Complete for $BASE_ID ---"
