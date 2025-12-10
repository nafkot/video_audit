# syncmusic

A CLI tool for robust music detection in short videos, with voice-over resistance.

## Features

- Detects music in videos (15-70 seconds) using multi-stage analysis
- Supports both instrumental music and music with vocals
- Resistant to loud voice-overs via AI source separation (Demucs)
- Concurrent processing of multiple videos
- Outputs detailed JSON results including similarity scores and volume analysis

## Installation

### Prerequisites

**System dependencies:**
```bash
# macOS
brew install ffmpeg chromaprint

# Ubuntu/Debian
sudo apt install ffmpeg libchromaprint-tools

# Other systems: install ffmpeg and fpcalc (Chromaprint)
```

**Python dependencies:**
```bash
cd python
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Build the CLI:**
```bash
go build -o syncmusic .
```

## Usage

### Basic Usage

```bash
./syncmusic --video="path/to/video.mp4" --audio="path/to/audio.mp3"
```

### Multiple Videos

```bash
./syncmusic --video="video1.mp4,video2.mp4,video3.mp4" --audio="audio.mp3"
```

### Command-line Flags

- `--video` (required): Comma-separated list of video file paths
- `--audio` (required): Path to target audio file
- `--threads` (optional): Number of concurrent threads (default: number of CPUs)
- `--robust` (optional): Enable Demucs + DTW stage for voice-over resistance (default: true)
- `--vocals` (optional): Target audio has vocals/singer, not instrumental (default: false)
  - **Important**: Requires exact same version of the song (no remixes, covers, or different mixes)
- `--threshold` (optional): Similarity threshold (0-100) to mark found as true (default: 70)
- `--verbose` (optional): Enable verbose error logging (default: false)

### Examples

**Detect instrumental music in videos:**
```bash
./syncmusic \
  --video="storage/video1.mp4,storage/video2.mp4" \
  --audio="storage/instrumental.mp3" \
  --threads=4 \
  --threshold=70
```

**Detect music with vocals (singer) in videos:**
```bash
./syncmusic \
  --video="storage/video1.mp4,storage/video2.mp4" \
  --audio="storage/song_with_vocals.mp3" \
  --vocals \
  --threads=4 \
  --threshold=70
```
> **Note**: The `--vocals` flag requires the target audio to be the exact same version as in the video. Remixes, covers, live versions, or differently mixed versions will not be detected reliably.

**Fast mode (disable voice-over resistance):**
```bash
./syncmusic \
  --video="storage/video.mp4" \
  --audio="storage/audio.mp3" \
  --robust=false
```

## Output

The CLI outputs a JSON array with one result per video:

```json
[
  {
    "video": "storage/video1.mp4",
    "found": true,
    "similarity": 85,
    "volume": 65
  },
  {
    "video": "storage/video2.mp4",
    "found": false,
    "similarity": 0
  }
]
```

### Output Fields

- `video`: Path to the video file
- `found`: Whether the audio was detected (similarity >= threshold)
- `similarity`: Similarity score (0-100)
- `volume`: Music volume level in video (0-100, only shown if found=true)
