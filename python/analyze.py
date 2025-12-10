#!/usr/bin/env python3

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import librosa
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# Increased warnings filter to cover all necessary modules
warnings.filterwarnings('ignore')

print(f"DEBUG: Running with interpreter: {sys.executable}", file=sys.stderr)
#sys.exit(1)
# Storage configuration
# FIX 1: Set REPORTS_ROOT for the persistent output folder
REPORTS_ROOT = Path(__file__).parent.parent / 'storage' / 'reports'
STORAGE_ROOT = Path(__file__).parent.parent / 'storage' / 'audio'
DETECT_AI_SCRIPT = Path(__file__).parent / 'detect_ai_audio.py'
ANALYZE_VOCALS_SCRIPT = Path(__file__).parent / 'analyze_vocals.py'


def main():
    parser = argparse.ArgumentParser(description='Analyze video audio for music detection')
    parser.add_argument('--video-audio', required=True, help='Path to video audio WAV')
    parser.add_argument('--video-id', required=True, help='The ID of the video file (e.g., JQF7GzmtSEY)')
    parser.add_argument('--target-audio', required=False, help='Path to target audio file (optional)')
    parser.add_argument('--threshold', type=int, default=70, help='Similarity threshold')
    parser.add_argument('--robust', action='store_true', help='Enable Demucs + DTW stage')
    parser.add_argument('--vocals', action='store_true', help='Target audio has vocals (not instrumental)')
    parser.add_argument('--output-file', required=True, help='Path to save the audio analysis JSON output (Go temp file)')
    args = parser.parse_args()

    # Create the reports folder if it doesn't exist
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Construct the final persistent output path for the user
    final_output_path = REPORTS_ROOT / f"{args.video_id}_audio_analysis.json"

    result = analyze_audio(
        args.video_audio,
        args.target_audio,
        args.video_id,
        args.threshold,
        args.robust,
        args.vocals
    )

    # --- FIX 2: Save the result to the PERSISTENT reports file ---
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # --- Save result to the TEMPORARY path specified by Go for communication ---
    # Go reads this file to get the result back, then deletes it.
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # This line prints a summary status to stderr (Go captures this but ignores stdout JSON)
    print(json.dumps({"status": "success", "output_file": str(final_output_path), "result_summary": result.get("folder", "N/A")}))


def analyze_audio(video_wav_path, target_audio_path, video_id, threshold, robust, vocals):
    """Main analysis pipeline"""

    # Get video duration
    video_duration = get_audio_duration(video_wav_path)

    # Check if audio matching should be performed
    skip_audio_matching = target_audio_path is None

    if skip_audio_matching:
        print(f"[DEBUG] No target audio provided - skipping audio matching", file=sys.stderr)
    else:
        print(f"[DEBUG] Video duration: {video_duration:.2f}s, Target audio duration: {get_audio_duration(target_audio_path):.2f}s", file=sys.stderr)

    # Initialize audio matching results (only used if target audio is provided)
    final_similarity = 0
    low_volume_percent = 0
    detected_low_volume = False
    detected_duration = 0
    audio_track_duration = 0

    # Perform audio matching only if target audio is provided
    if not skip_audio_matching:
        audio_track_duration = get_audio_duration(target_audio_path)

        # Stage A: Chroma matching (fast)
        stage_a_similarity, stage_a_match_duration = chromaprint_match(video_wav_path, target_audio_path)
        print(f"[DEBUG] Stage A similarity: {stage_a_similarity}", file=sys.stderr)

        final_similarity = stage_a_similarity
        detected_duration = stage_a_match_duration

        # Stage B: Demucs + DTW (only if robust mode is enabled)
        if robust:
            try:
                stage_b_similarity, low_volume_percent, stage_b_match_duration = demucs_dtw_match(
                    video_wav_path,
                    target_audio_path,
                    vocals
                )

                print(f"[DEBUG] Stage B similarity: {stage_b_similarity}, low_volume: {low_volume_percent}%", file=sys.stderr)

                # Use the higher similarity
                if stage_b_similarity > stage_a_similarity:
                    final_similarity = stage_b_similarity
                    detected_duration = stage_b_match_duration

                detected_low_volume = low_volume_percent >= 50

            except Exception as e:
                print(f"Warning: Stage B failed, using Stage A only: {e}", file=sys.stderr)

    # Run robust analysis (AI detection + vocals analysis) if requested
    ai_result = {"ai_generated": False, "ai_confidence": 0}
    vocals_analysis = {}

    if robust:
        # Separate audio sources (needed for vocals analysis and potentially for AI detection)
        if skip_audio_matching:
            # Only separate if we haven't already done it in demucs_dtw_match
            try:
                print(f"[DEBUG] Separating audio sources for vocals analysis", file=sys.stderr)
                separate_sources(video_wav_path)
            except Exception as e:
                print(f"Warning: Audio separation failed: {e}", file=sys.stderr)

        # AI detection on ORIGINAL audio (not separated stems)
        try:
            ai_result = detect_ai_generated_audio_original(video_wav_path)
        except Exception as e:
            print(f"Warning: AI detection failed: {e}", file=sys.stderr)

        # Vocals analysis using Whisper
        try:
            vocals_analysis = analyze_vocals_whisper(video_wav_path)
        except Exception as e:
            print(f"Warning: Vocals analysis failed: {e}", file=sys.stderr)

    # Build result dictionary
    result = {
        "video": str(video_wav_path),
        "ai_generated": bool(ai_result["ai_generated"]),
        "ai_confidence": int(ai_result["ai_confidence"]),
        "folder": video_id, # FIX 3: Ensure this always uses the clean video_id
        "vocals_analysis": vocals_analysis,
        "video_duration": round(video_duration, 2),
    }

    # Add audio matching results only if target audio was provided
    if not skip_audio_matching:
        result.update({
            "found": bool(final_similarity >= threshold),
            "similarity": int(final_similarity),
            "detected_low_volume": bool(detected_low_volume),
            "low_volume_percent": int(low_volume_percent),
            "audio_track_duration": round(audio_track_duration, 2),
            "detected_duration": round(detected_duration, 2) if detected_duration > 0 else 0
        })

    return result


def chromaprint_match(video_wav, target_audio):
# (function body remains unchanged)
    """Stage A: Chroma-based cross-correlation matching"""
    try:
        # Load audio files
        video_audio, sr = librosa.load(video_wav, sr=22050, mono=True)
        target_audio_data, sr = librosa.load(target_audio, sr=22050, mono=True)

        # Compute chroma features
        video_chroma = librosa.feature.chroma_cqt(y=video_audio, sr=sr, hop_length=512)
        target_chroma = librosa.feature.chroma_cqt(y=target_audio_data, sr=sr, hop_length=512)

        # Normalize
        video_chroma = librosa.util.normalize(video_chroma, axis=0)
        target_chroma = librosa.util.normalize(target_chroma, axis=0)

        # Use sliding window cross-correlation
        similarity, match_frames = sliding_chroma_match(video_chroma, target_chroma)

        # Convert frames to seconds
        hop_length = 512
        match_duration = (match_frames * hop_length) / sr

        print(f"[DEBUG] Match duration: {match_duration:.2f}s ({match_frames} frames)", file=sys.stderr)

        return similarity, match_duration

    except Exception as e:
        print(f"Chroma matching failed: {e}", file=sys.stderr)
        return 0, 0


def sliding_chroma_match(video_chroma, target_chroma):
# (function body remains unchanged)
    """Slide target chroma over video chroma and find best match"""

    video_len = video_chroma.shape[1]
    target_len = target_chroma.shape[1]

    if target_len > video_len:
        # Target is longer than video, swap
        video_chroma, target_chroma = target_chroma, video_chroma
        video_len, target_len = target_len, video_len

    best_similarity = -1
    match_frames = target_len  # Default to target length
    step_size = max(1, target_len // 10)  # Sample fewer positions for speed

    # Slide target over video
    for i in range(0, max(1, video_len - target_len + 1), step_size):
        end_idx = min(i + target_len, video_len)
        video_segment = video_chroma[:, i:end_idx]
        target_segment = target_chroma[:, :video_segment.shape[1]]

        # Use correlation-based similarity (more strict)
        if video_segment.shape[1] == target_segment.shape[1] and video_segment.shape[1] > 0:
            # Flatten and compute correlation
            v_flat = video_segment.flatten()
            t_flat = target_segment.flatten()

            # Pearson correlation
            correlation = np.corrcoef(v_flat, t_flat)[0, 1]
            if not np.isnan(correlation):
                if correlation > best_similarity:
                    best_similarity = correlation
                    match_frames = video_segment.shape[1]

    print(f"[DEBUG] Best correlation: {best_similarity:.3f}", file=sys.stderr)

    # Convert to 0-100 scale
    # Correlation ranges from -1 to 1
    # For matching audio excerpts/snippets, correlation is typically 0.2-0.6
    # Map 0.2 -> 0%, 0.6 -> 100% (with extrapolation beyond)
    if best_similarity < 0.2:
        similarity_percent = 0
    else:
        similarity_percent = min(100, (best_similarity - 0.2) / 0.4 * 100)

    return similarity_percent, match_frames


def demucs_dtw_match(video_wav, target_audio, vocals=False):
# (function body remains unchanged)
    """Stage B: Demucs source separation + DTW chroma matching"""

    # Always separate video audio to remove voice-over
    video_stems = separate_sources(video_wav)
    video_acc = build_accompaniment(video_stems)

    if vocals:
        # Target has vocals - separate it to get instrumental part
        target_stems = separate_sources(target_audio)
        target_acc = build_accompaniment(target_stems)
    else:
        # Target is already instrumental - use as-is
        target_acc, _ = librosa.load(target_audio, sr=44100, mono=True)

    # Compute chroma features
    video_chroma = compute_chroma_cens(video_acc)
    target_chroma = compute_chroma_cens(target_acc)

    # Use sliding correlation matching (same as Stage A but on separated audio)
    similarity, match_frames = sliding_chroma_match(video_chroma, target_chroma)

    # Convert frames to seconds (hop_length=512 used in compute_chroma_cens)
    hop_length = 512
    sr = 44100
    match_duration = (match_frames * hop_length) / sr

    print(f"[DEBUG] Stage B match duration: {match_duration:.2f}s ({match_frames} frames)", file=sys.stderr)

    # For volume estimation, use DTW to find alignment
    _, path = fastdtw(target_chroma.T, video_chroma.T, dist=euclidean)

    # Estimate volume difference using aligned segments
    low_volume_percent = estimate_volume_difference(
        video_acc, target_acc, path
    )

    return similarity, low_volume_percent, match_duration


def get_audio_id(audio_path):
# (function body remains unchanged)
    """Generate unique ID for audio file based on its path"""
    return hashlib.md5(str(audio_path).encode()).hexdigest()[:16]


def get_audio_duration(audio_path):
# (function body remains unchanged)
    """Get duration of audio file in seconds using librosa"""
    try:
        duration = librosa.get_duration(path=audio_path)
        print(f"[DEBUG] Audio duration for {Path(audio_path).name}: {duration:.2f}s", file=sys.stderr)
        return duration
    except Exception as e:
        print(f"[ERROR] Failed to get audio duration for {audio_path}: {e}", file=sys.stderr)
        return 0.0


def get_storage_path(audio_path):
# (function body remains unchanged)
    """Get storage path for separated audio stems"""
    audio_hash = hashlib.md5(str(audio_path).encode()).hexdigest()[:16]
    storage_path = STORAGE_ROOT / audio_hash
    return storage_path


def separate_sources(audio_path):
# (function body remains unchanged)
    """Separate audio into stems using Demucs"""
    storage_path = get_storage_path(audio_path)

    # Check if already separated
    vocals_path = storage_path / 'vocals.wav'
    instrumental_path = storage_path / 'instrumental.wav'

    if vocals_path.exists() and instrumental_path.exists():
        print(f"[DEBUG] Using cached separated audio from: {storage_path}", file=sys.stderr)
        return {
            'vocals': vocals_path,
            'no_vocals': instrumental_path,
        }

    # Create storage directory
    storage_path.mkdir(parents=True, exist_ok=True)

    # Use temporary directory for Demucs processing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Run Demucs using the current Python interpreter (venv-aware)
        subprocess.run(
            [
                sys.executable, '-m', 'demucs',
                '--two-stems', 'vocals',  # Faster: only split vocals vs rest
                '-o', str(tmpdir_path),
                '-n', 'htdemucs',
                audio_path
            ],
            capture_output=True,
            timeout=120
        )

        # Find output stems
        # Demucs creates: tmpdir/htdemucs/filename/{vocals,no_vocals}.wav
        audio_name = Path(audio_path).stem
        stem_dir = tmpdir_path / 'htdemucs' / audio_name

        temp_vocals = stem_dir / 'vocals.wav'
        temp_instrumental = stem_dir / 'no_vocals.wav'

        # Copy to persistent storage
        if temp_vocals.exists():
            shutil.copy2(temp_vocals, vocals_path)
            print(f"[DEBUG] Saved vocals to: {vocals_path}", file=sys.stderr)

        if temp_instrumental.exists():
            shutil.copy2(temp_instrumental, instrumental_path)
            print(f"[DEBUG] Saved instrumental to: {instrumental_path}", file=sys.stderr)

    stems = {
        'vocals': vocals_path,
        'no_vocals': instrumental_path,
    }

    return stems


def build_accompaniment(stems):
# (function body remains unchanged)
    """Build accompaniment track from non-vocal stems"""
    # With --two-stems vocals, we get vocals and no_vocals
    # Use no_vocals as accompaniment

    if stems['no_vocals'].exists():
        print(f"[DEBUG] Loading no_vocals from: {stems['no_vocals']}", file=sys.stderr)
        audio, _ = librosa.load(stems['no_vocals'], sr=44100, mono=True)
        print(f"[DEBUG] Loaded audio shape: {audio.shape}, energy: {np.sqrt(np.mean(audio**2)):.6f}", file=sys.stderr)
        return audio
    else:
        # Fallback: return empty audio
        print(f"[DEBUG] WARNING: no_vocals file not found at {stems['no_vocals']}, returning silence!", file=sys.stderr)
        return np.zeros(44100)


def compute_chroma_cens(audio, sr=44100):
# (function body remains unchanged)
    """Compute chroma CENS features"""
    # Chroma CENS is robust to dynamics and timbre variations
    chroma = librosa.feature.chroma_cens(
        y=audio,
        sr=sr,
        hop_length=512,
        n_chroma=12
    )

    # Normalize
    chroma = librosa.util.normalize(chroma, axis=0)

    return chroma


def estimate_volume_difference(video_acc, target_acc, dtw_path):
# (function body remains unchanged)
    """Estimate volume difference from DTW alignment"""

    # Extract aligned frames
    target_frames = [p[0] for p in dtw_path]
    video_frames = [p[1] for p in dtw_path]

    # Convert frame indices to sample indices (hop_length=512)
    hop_length = 512
    target_samples = [f * hop_length for f in target_frames]
    video_samples = [f * hop_length for f in video_frames]

    # Compute RMS energy for aligned segments
    target_rms_values = []
    video_rms_values = []

    for t_idx, v_idx in zip(target_samples[:100], video_samples[:100]):  # Sample first 100 frames
        # Extract small windows
        t_window = target_acc[t_idx:t_idx+hop_length]
        v_window = video_acc[v_idx:v_idx+hop_length]

        if len(t_window) > 0 and len(v_window) > 0:
            target_rms_values.append(np.sqrt(np.mean(t_window**2)))
            video_rms_values.append(np.sqrt(np.mean(v_window**2)))

    if not target_rms_values or not video_rms_values:
        return 0

    # Compute amplitude ratio
    target_rms = np.mean(target_rms_values)
    video_rms = np.mean(video_rms_values)

    if target_rms == 0 or video_rms == 0:
        return 0

    # Ratio of video to target (if video is quieter, ratio < 1)
    ratio = video_rms / target_rms

    # Convert to dB
    db = 20 * np.log10(ratio) if ratio > 0 else -40

    # Map to 0-100 scale: -40 dB -> 100%, 0 dB -> 0%
    low_volume_percent = max(0, min(100, (-min(db, 0) / 40) * 100))

    return low_volume_percent


def detect_ai_generated_audio_original(audio_path):
# (function body remains unchanged)
    """
    Detect if ORIGINAL audio (before Demucs separation) is AI-generated
    """
    try:
        # Run AI detection on ORIGINAL audio
        result = subprocess.run(
            [sys.executable, str(DETECT_AI_SCRIPT), '--audio', str(audio_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            detection = json.loads(result.stdout)
            print(f"[DEBUG] AI detection on original audio: ai_generated={detection['ai_generated']}, confidence={detection['ai_confidence']}", file=sys.stderr)
            return detection
        else:
            print(f"[DEBUG] AI detection failed: {result.stderr}", file=sys.stderr)
            return {"ai_generated": False, "ai_confidence": 0}

    except Exception as e:
        print(f"[DEBUG] Error detecting AI: {e}", file=sys.stderr)
        return {"ai_generated": False, "ai_confidence": 0}


def analyze_vocals_whisper(audio_path):
# (function body remains unchanged)
    """
    Analyze vocals using Whisper (transcription, sentiment, topic classification)
    """
    storage_path = get_storage_path(audio_path)
    vocals_path = storage_path / 'vocals.wav'

    if not vocals_path.exists():
        print(f"[DEBUG] Vocals file not found for Whisper analysis", file=sys.stderr)
        return {}

    try:
        # Run Whisper analysis script (using 'small' model for better translation quality)
        result = subprocess.run(
            [sys.executable, str(ANALYZE_VOCALS_SCRIPT), '--audio', str(vocals_path), '--model', 'small'],
            capture_output=True,
            text=True,
            timeout=300  # Increased timeout for small model (5 minutes)
        )

        if result.returncode == 0:
            analysis = json.loads(result.stdout)
            print(f"[DEBUG] Vocals analysis: language={analysis.get('language', 'unknown')}, topic={analysis.get('topic', {}).get('category', 'unknown')}", file=sys.stderr)
            return analysis
        else:
            print(f"[DEBUG] Vocals analysis failed: {result.stderr}", file=sys.stderr)
            return {}

    except Exception as e:
        print(f"[DEBUG] Error analyzing vocals: {e}", file=sys.stderr)
        return {}


def separate_sources(audio_path):
# (function body remains unchanged)
    """Separate audio into stems using Demucs"""
    storage_path = get_storage_path(audio_path)

    # Check if already separated
    vocals_path = storage_path / 'vocals.wav'
    instrumental_path = storage_path / 'no_vocals.wav'

    if vocals_path.exists() and instrumental_path.exists():
        print(f"[DEBUG] Using cached separated audio from: {storage_path}", file=sys.stderr)
        return {
            'vocals': vocals_path,
            'no_vocals': instrumental_path,
        }

    # Create storage directory
    storage_path.mkdir(parents=True, exist_ok=True)

    # Use temporary directory for Demucs processing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Run Demucs using the current Python interpreter (venv-aware)
        subprocess.run(
            [
                sys.executable, '-m', 'demucs',
                '--two-stems', 'vocals',  # Faster: only split vocals vs rest
                '-o', str(tmpdir_path),
                '-n', 'htdemucs',
                audio_path
            ],
            capture_output=True,
            timeout=120
        )

        # Find output stems
        # Demucs creates: tmpdir/htdemucs/filename/{vocals,no_vocals}.wav
        audio_name = Path(audio_path).stem
        stem_dir = tmpdir_path / 'htdemucs' / audio_name

        temp_vocals = stem_dir / 'vocals.wav'
        temp_instrumental = stem_dir / 'no_vocals.wav'

        # Copy to persistent storage
        if temp_vocals.exists():
            shutil.copy2(temp_vocals, vocals_path)
            print(f"[DEBUG] Saved vocals to: {vocals_path}", file=sys.stderr)

        if temp_instrumental.exists():
            shutil.copy2(temp_instrumental, instrumental_path)
            print(f"[DEBUG] Saved instrumental to: {instrumental_path}", file=sys.stderr)

    stems = {
        'vocals': vocals_path,
        'no_vocals': instrumental_path,
    }

    return stems


if __name__ == '__main__':
    main()
