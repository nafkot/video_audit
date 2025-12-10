#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings('ignore')

# Try to import transformers for HuggingFace model
try:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    import torch
    import librosa
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[ERROR] transformers/torch not installed. Install with: pip install transformers torch", file=sys.stderr)

# Model configuration
HUGGINGFACE_MODEL = "MelodyMachine/Deepfake-audio-detection-V2"
SAMPLE_RATE = 16000  # wav2vec2 expects 16kHz

# Global cache for model
_model_cache = None
_feature_extractor_cache = None


def main():
    parser = argparse.ArgumentParser(description='Detect if audio/video is AI-generated')
    parser.add_argument('--audio', help='Path to audio file')
    parser.add_argument('--video', help='Path to video file')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    if not TRANSFORMERS_AVAILABLE:
        print(json.dumps({
            "ai_generated": False,
            "ai_confidence": 0,
            "error": "transformers library not installed"
        }))
        sys.exit(1)

    # Determine input type
    if args.video:
        audio_path = extract_audio_from_video(args.video)
        is_temp = True
        source = args.video
    elif args.audio:
        audio_path = args.audio
        is_temp = False
        source = args.audio
    else:
        print("[ERROR] Must provide either --audio or --video", file=sys.stderr)
        sys.exit(1)

    if not audio_path or not os.path.exists(audio_path):
        print(json.dumps({
            "source": source,
            "ai_generated": False,
            "ai_confidence": 0,
            "error": "Failed to load audio"
        }))
        sys.exit(1)

    try:
        # Detect AI
        detection = detect_ai_audio(audio_path)

        result = {
            "source": source,
            "ai_generated": detection["ai_generated"],
            "ai_confidence": detection["ai_confidence"]
        }

        if "detection_details" in detection:
            result["details"] = detection["detection_details"]

        if args.json or args.video:
            # Always JSON for video or when explicitly requested
            print(json.dumps(result))
        else:
            # Human-readable output for audio
            status = "AI-GENERATED" if result["ai_generated"] else "REAL/HUMAN"
            print(f"\n{'='*60}")
            print(f"Audio: {source}")
            print(f"Status: {status}")
            print(f"AI Confidence: {result['ai_confidence']}%")
            print(f"{'='*60}")

    finally:
        # Cleanup temp audio file if it was extracted from video
        if is_temp and audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


def extract_audio_from_video(video_path):
    """Extract audio from video as WAV file"""
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
        return None

    try:
        # Create temporary WAV file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='video_audio_')
        os.close(temp_fd)

        print(f"[DEBUG] Extracting audio from video: {video_path}", file=sys.stderr)

        # Use ffmpeg to extract audio as 16kHz mono WAV
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(SAMPLE_RATE),  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            temp_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"[ERROR] FFmpeg failed: {result.stderr.decode()}", file=sys.stderr)
            os.remove(temp_path)
            return None

        print(f"[DEBUG] Audio extracted to: {temp_path}", file=sys.stderr)
        return temp_path

    except Exception as e:
        print(f"[ERROR] Failed to extract audio: {e}", file=sys.stderr)
        return None


def load_model():
    """Load pretrained model and feature extractor (cached)"""
    global _model_cache, _feature_extractor_cache

    if _model_cache is not None and _feature_extractor_cache is not None:
        return _feature_extractor_cache, _model_cache

    print(f"[DEBUG] Loading model {HUGGINGFACE_MODEL}...", file=sys.stderr)

    try:
        # Load feature extractor and model
        feature_extractor = AutoFeatureExtractor.from_pretrained(HUGGINGFACE_MODEL)
        model = AutoModelForAudioClassification.from_pretrained(HUGGINGFACE_MODEL)

        # Set to evaluation mode
        model.eval()

        # Cache for future use
        _feature_extractor_cache = feature_extractor
        _model_cache = model

        print(f"[DEBUG] Model loaded successfully", file=sys.stderr)
        return feature_extractor, model

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}", file=sys.stderr)
        raise


def detect_ai_audio(audio_path):
    """
    Detect if audio is AI-generated using MelodyMachine/Deepfake-audio-detection-V2

    Returns:
        dict: {
            "ai_generated": bool,
            "ai_confidence": int (0-100),
            "detection_details": dict (optional)
        }
    """
    try:
        # Load model
        feature_extractor, model = load_model()

        # Load audio file at 16kHz (wav2vec2 requirement)
        print(f"[DEBUG] Loading audio: {audio_path}", file=sys.stderr)
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        # Check if audio is too short (minimum 0.5 seconds)
        if len(audio) < SAMPLE_RATE * 0.5:
            print(f"[WARNING] Audio too short: {len(audio)/SAMPLE_RATE:.2f}s", file=sys.stderr)
            return {
                "ai_generated": False,
                "ai_confidence": 0,
                "error": "audio_too_short"
            }

        # Prepare input for model
        inputs = feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        # Run inference
        print(f"[DEBUG] Running inference...", file=sys.stderr)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get prediction
        predicted_class_idx = torch.argmax(logits, dim=-1).item()

        # Get confidence scores (softmax probabilities)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        confidence_scores = probabilities.tolist()

        # Model labels (typically: 0=real, 1=fake or vice versa)
        id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: "real", 1: "fake"}

        predicted_label = id2label.get(predicted_class_idx, "unknown")
        confidence = confidence_scores[predicted_class_idx] * 100

        # Determine if AI generated
        is_ai_generated = predicted_label.lower() in ["fake", "ai", "synthetic", "deepfake", "label_1", "1"]

        # If prediction is real/human, invert confidence for clarity
        if not is_ai_generated:
            ai_confidence = 100 - confidence
        else:
            ai_confidence = confidence

        print(f"[DEBUG] Prediction: {predicted_label}, confidence: {confidence:.2f}%, AI confidence: {ai_confidence:.2f}%", file=sys.stderr)

        return {
            "ai_generated": bool(ai_confidence >= 50),  # Threshold: 50%
            "ai_confidence": int(ai_confidence),
            "detection_details": {
                "model_prediction": predicted_label,
                "raw_confidence": float(confidence),
                "all_scores": {id2label.get(i, f"label_{i}"): float(score * 100) for i, score in enumerate(confidence_scores)}
            }
        }

    except Exception as e:
        print(f"[ERROR] Detection failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

        return {
            "ai_generated": False,
            "ai_confidence": 0,
            "error": str(e)
        }


if __name__ == '__main__':
    main()
