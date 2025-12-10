#!/usr/bin/env python3

import argparse
import json
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

warnings.filterwarnings('ignore')

# Try to import required libraries
try:
    import whisper
    from textblob import TextBlob
    from transformers import pipeline
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    LIBRARIES_AVAILABLE = False
    print(f"[ERROR] Required libraries not installed: {e}", file=sys.stderr)
    print("[ERROR] Install with: pip install openai-whisper textblob transformers", file=sys.stderr)

@lru_cache(maxsize=1)
def _load_profanity_list() -> set[str]:
# (function body remains unchanged)
    # Get script directory and construct path to profanity list
    script_dir = Path(__file__).resolve().parent
    profanity_file = script_dir.parent / "profanity-list.txt"

    try:
        print(f"[DEBUG] Loading profanity list from: {profanity_file}", file=sys.stderr)

        with profanity_file.open('r', encoding='utf-8') as f:
            # Read all lines, strip whitespace, convert to lowercase, filter empty lines
            words = {
                line.strip().lower()
                for line in f
                if line.strip()  # Skip empty lines
            }

        print(f"[DEBUG] Loaded {len(words)} profanity words", file=sys.stderr)
        return words

    except FileNotFoundError:
        print(f"[ERROR] Profanity list not found at: {profanity_file}", file=sys.stderr)
        print(f"[ERROR] Using empty profanity list as fallback", file=sys.stderr)
        return set()  # Return empty set as fallback

    except IOError as e:
        print(f"[ERROR] Failed to read profanity list: {e}", file=sys.stderr)
        print(f"[ERROR] Using empty profanity list as fallback", file=sys.stderr)
        return set()  # Return empty set as fallback

# Global cache for models
_whisper_model = None
_sentiment_pipeline = None
_classifier_pipeline = None


def main():
# (function body remains unchanged)
    parser = argparse.ArgumentParser(description='Analyze vocals audio with Whisper')
    parser.add_argument('--audio', required=True, help='Path to vocals audio file')
    parser.add_argument('--model', default='small', help='Whisper model size (tiny, base, small, medium, large)')
    args = parser.parse_args()

    if not LIBRARIES_AVAILABLE:
        print(json.dumps({
            "error": "Required libraries not installed",
            "language": "",
            "transcript": "",
            "sentiment": {},
            "topic": {},
            "offensive_content": False,
            "offensive_words": []
        }))
        sys.exit(1)

    if not os.path.exists(args.audio):
        print(json.dumps({
            "error": "Audio file not found",
            "language": "",
            "transcript": "",
            "sentiment": {},
            "topic": {},
            "offensive_content": False,
            "offensive_words": []
        }))
        sys.exit(1)

    result = analyze_vocals(args.audio, args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def analyze_vocals(audio_path, model_size='base'):
# (function body remains unchanged)
    try:
        # Step 1: Transcribe with Whisper
        print(f"[DEBUG] Loading Whisper model: {model_size}", file=sys.stderr)
        transcript_data = transcribe_audio(audio_path, model_size)

        language = transcript_data.get('language', 'unknown')
        transcript = transcript_data.get('text', '').strip()
        transcript_english = transcript_data.get('text_english', '').strip()

        print(f"[DEBUG] Detected language: {language}", file=sys.stderr)
        print(f"[DEBUG] Transcript length: {len(transcript)} chars", file=sys.stderr)
        if language != "en":
            print(f"[DEBUG] English translation for analysis: {transcript_english[:100]}...", file=sys.stderr)

        if not transcript:
            return {
                "language": language,
                "transcript": "",
                "sentiment": {"polarity": 0, "subjectivity": 0, "label": "neutral"},
                "topic": {"category": "unknown", "confidence": 0},
                "offensive_content": False,
                "offensive_words": []
            }

        # Step 2: Sentiment Analysis (use English translation)
        sentiment = analyze_sentiment(transcript_english)

        # Step 3: Topic/Intent Classification (use English translation)
        topic = classify_topic(transcript_english)

        # Step 4: Offensive Content Detection (use English translation)
        offensive_words = detect_offensive_words(transcript_english)
        offensive_content = len(offensive_words) > 0

        return {
            "language": language,
            "transcript": transcript,  # Original language for JSON output
            "transcript_english": transcript_english, # Retain English for narrative_analyser use
            "sentiment": sentiment,
            "topic": topic,
            "offensive_content": offensive_content,
            "offensive_words": offensive_words
        }

    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

        return {
            "error": str(e),
            "language": "",
            "transcript": "",
            "sentiment": {},
            "topic": {},
            "offensive_content": False,
            "offensive_words": []
        }


def transcribe_audio(audio_path, model_size='base'):
    """Transcribe audio using Whisper (original + English translation)"""
    global _whisper_model

    try:
        # Load model (cached)
        if _whisper_model is None or _whisper_model != model_size:
            print(f"[DEBUG] Loading Whisper model: {model_size}", file=sys.stderr)
            _whisper_model = whisper.load_model(model_size)

        # Transcribe in original language
        print(f"[DEBUG] Transcribing audio...", file=sys.stderr)
        result = _whisper_model.transcribe(audio_path)

        original_text = result.get("text", "")
        language = result.get("language", "unknown")

        # --- MODIFIED: Ensure english_text is always available ---
        english_text = original_text
        if language != "en" and original_text:
            print(f"[DEBUG] Translating to English for analysis...", file=sys.stderr)
            translation_result = _whisper_model.transcribe(audio_path, task="translate")
            english_text = translation_result.get("text", original_text)
        # --- END MODIFIED ---

        return {
            "language": language,
            "text": original_text,
            "text_english": english_text
        }

    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}", file=sys.stderr)
        return {"language": "unknown", "text": "", "text_english": ""}


def analyze_sentiment(text):
# (function body remains unchanged)
    """
    Analyze sentiment using TextBlob

    Returns:
        dict: {
            "polarity": float (-1 to 1),
            "subjectivity": float (0 to 1),
            "label": str (positive/negative/neutral)
        }
    """
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Classify sentiment
        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"

        print(f"[DEBUG] Sentiment: {label} (polarity={polarity:.2f}, subjectivity={subjectivity:.2f})", file=sys.stderr)

        return {
            "polarity": round(float(polarity), 3),
            "subjectivity": round(float(subjectivity), 3),
            "label": label
        }

    except Exception as e:
        print(f"[ERROR] Sentiment analysis failed: {e}", file=sys.stderr)
        return {"polarity": 0, "subjectivity": 0, "label": "neutral"}


def classify_topic(text):
# (function body remains unchanged)
    """
    HYBRID topic classification using keywords + ML model

    Returns: dict with category, confidence, and all scores
    """
    global _classifier_pipeline

    try:
        # Step 1: Keyword-based detection (fast)
        keyword_result = classify_by_keywords(text)

        # Step 2: ML-based classification (comprehensive)
        ml_result = classify_by_ml(text)

        # Step 3: Combine results for best accuracy
        final_result = combine_classification_results(keyword_result, ml_result)

        print(f"[DEBUG] Topic: {final_result['category']} (confidence={final_result['confidence']:.2f}, method={final_result['detection_method']})", file=sys.stderr)

        return final_result

    except Exception as e:
        print(f"[ERROR] Topic classification failed: {e}", file=sys.stderr)
        return {
            "category": "Other",
            "confidence": 0,
            "secondary_categories": [],
            "detection_method": "error"
        }


def classify_by_keywords(text):
# (function body remains unchanged)
    """
    Fast keyword-based classification

    Returns: dict with category, confidence, and matched_keywords
    """
    text_lower = text.lower()

    # Keyword patterns for each category
    KEYWORD_PATTERNS = {
        "Product Review": {
            "keywords": ["review", "unbox", "test", "rating", "recommend", "pros", "cons", "worth it", "purchase", "buy", "quality"],
            "weight": 1.0
        },
        "Tutorial": {
            "keywords": ["how to", "step", "learn", "tutorial", "guide", "show you", "teach", "lesson", "demonstration", "explain"],
            "weight": 1.0
        },
        "Spam": {
            "keywords": ["buy now", "click here", "limited", "offer", "discount", "free", "act now", "exclusive", "promotion", "deal"],
            "weight": 1.2  # Higher weight because spam is dangerous
        },
        "Storytime": {
            "keywords": ["story", "happened", "once", "day", "experience", "told", "remember when", "time when", "back when"],
            "weight": 1.0
        },
        "Personal/Vlog": {
            "keywords": ["my day", "today i", "i went", "vlog", "my life", "diary", "personal", "my experience"],
            "weight": 1.0
        },
        "Comedy/Humor": {
            "keywords": ["funny", "joke", "laugh", "hilarious", "haha", "comedy", "humor", "prank", "ridiculous"],
            "weight": 1.0
        },
        "Motivational": {
            "keywords": ["inspire", "motivate", "success", "achieve", "goal", "dream", "believe", "power", "strength"],
            "weight": 1.0
        },
        "Complaint/Rant": {
            "keywords": ["angry", "frustrated", "terrible", "worst", "hate", "complaint", "disappointed", "annoying"],
            "weight": 1.0
        },
        "Political Commentary": {
            "keywords": ["government", "election", "vote", "policy", "president", "minister", "politics", "law", "democracy"],
            "weight": 1.0
        },
        "Cooking/Recipe": {
            "keywords": ["cook", "recipe", "ingredient", "bake", "kitchen", "delicious", "taste", "food", "prepare"],
            "weight": 1.0
        },
        "Gaming": {
            "keywords": ["game", "play", "level", "boss", "controller", "console", "gamer", "gameplay", "stream"],
            "weight": 1.0
        },
        "Music/Song": {
            "keywords": ["sing", "music", "song", "lyrics", "melody", "rhythm", "beat", "album", "artist"],
            "weight": 1.0
        }
    }

    # Calculate scores for each category
    category_scores = {}
    matched_keywords = {}

    for category, data in KEYWORD_PATTERNS.items():
        keywords = data["keywords"]
        weight = data["weight"]
        matches = []

        for keyword in keywords:
            if keyword in text_lower:
                matches.append(keyword)

        if matches:
            # Score based on number of matches and weight
            score = (len(matches) / len(keywords)) * weight
            category_scores[category] = score
            matched_keywords[category] = matches

    # Get top category
    if category_scores:
        top_category = max(category_scores, key=category_scores.get)
        confidence = min(0.85, category_scores[top_category])  # Cap at 0.85

        return {
            "category": top_category,
            "confidence": confidence,
            "matched_keywords": matched_keywords.get(top_category, []),
            "method": "keywords"
        }

    return None


def classify_by_ml(text):
# (function body remains unchanged)
    """
    ML-based zero-shot classification

    Returns: dict with category, confidence, and all scores
    """
    global _classifier_pipeline

    try:
        # Load classifier (cached)
        if _classifier_pipeline is None:
            print(f"[DEBUG] Loading zero-shot classifier...", file=sys.stderr)
            _classifier_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )

        # Extended categories (more specific)
        categories = [
            "product review or unboxing",
            "political commentary or debate",
            "tutorial or educational content",
            "spam or advertisement",
            "entertainment or comedy",
            "news or journalism",
            "personal vlog or diary",
            "music or song lyrics",
            "storytime or anecdote",
            "motivational speech",
            "complaint or rant",
            "cooking or recipe",
            "gaming content"
        ]

        # Classify (limit text to first 512 chars for speed)
        text_sample = text[:512]
        result = _classifier_pipeline(text_sample, categories)

        # Map to simpler category names
        category_map = {
            "product review or unboxing": "Product Review",
            "political commentary or debate": "Political Commentary",
            "tutorial or educational content": "Tutorial",
            "spam or advertisement": "Spam",
            "entertainment or comedy": "Comedy/Humor",
            "news or journalism": "News",
            "personal vlog or diary": "Personal/Vlog",
            "music or song lyrics": "Music/Song",
            "storytime or anecdote": "Storytime",
            "motivational speech": "Motivational",
            "complaint or rant": "Complaint/Rant",
            "cooking or recipe": "Cooking/Recipe",
            "gaming content": "Gaming"
        }

        top_category = result['labels'][0]
        top_confidence = result['scores'][0]

        # Get secondary categories (top 3)
        secondary = []
        for i in range(1, min(4, len(result['labels']))):
            if result['scores'][i] > 0.15:  # Only include if confidence > 15%
                secondary.append({
                    "category": category_map.get(result['labels'][i], result['labels'][i]),
                    "confidence": round(float(result['scores'][i]), 3)
                })

        return {
            "category": category_map.get(top_category, "Other"),
            "confidence": top_confidence,
            "secondary_categories": secondary,
            "method": "ml"
        }

    except Exception as e:
        print(f"[ERROR] ML classification failed: {e}", file=sys.stderr)
        return None


def combine_classification_results(keyword_result, ml_result):
# (function body remains unchanged)
    """
    Combine keyword and ML results for best accuracy

    Strategy:
    1. If both agree → boost confidence
    2. If keyword found but low ML confidence → use keyword
    3. If no keywords but high ML confidence → use ML
    4. If low confidence overall → mark as "Other"
    """
    # Case 1: No results from either method
    if not keyword_result and not ml_result:
        return {
            "category": "Other",
            "confidence": 0,
            "secondary_categories": [],
            "detection_method": "none"
        }

    # Case 2: Only keyword result
    if keyword_result and not ml_result:
        return {
            "category": keyword_result["category"],
            "confidence": round(keyword_result["confidence"], 3),
            "secondary_categories": [],
            "detection_method": "keywords_only",
            "matched_keywords": keyword_result.get("matched_keywords", [])
        }

    # Case 3: Only ML result
    if ml_result and not keyword_result:
        # Use ML but mark as "Other" if confidence too low
        if ml_result["confidence"] < 0.35:
            return {
                "category": "Other",
                "confidence": round(ml_result["confidence"], 3),
                "secondary_categories": ml_result.get("secondary_categories", []),
                "detection_method": "ml_low_confidence"
            }

        return {
            "category": ml_result["category"],
            "confidence": round(ml_result["confidence"], 3),
            "secondary_categories": ml_result.get("secondary_categories", []),
            "detection_method": "ml_only"
        }

    # Case 4: Both results available - COMBINE!
    keyword_cat = keyword_result["category"]
    ml_cat = ml_result["category"]

    # If they agree → boost confidence!
    if keyword_cat == ml_cat:
        boosted_confidence = min(0.95, keyword_result["confidence"] + ml_result["confidence"] * 0.3)
        return {
            "category": keyword_cat,
            "confidence": round(boosted_confidence, 3),
            "secondary_categories": ml_result.get("secondary_categories", []),
            "detection_method": "hybrid_agree",
            "matched_keywords": keyword_result.get("matched_keywords", [])
        }

    # If they disagree → use the one with higher confidence
    if keyword_result["confidence"] > ml_result["confidence"]:
        return {
            "category": keyword_cat,
            "confidence": round(keyword_result["confidence"], 3),
            "secondary_categories": [{"category": ml_cat, "confidence": round(ml_result["confidence"], 3)}],
            "detection_method": "hybrid_keyword_win",
            "matched_keywords": keyword_result.get("matched_keywords", [])
        }
    else:
        return {
            "category": ml_cat,
            "confidence": round(ml_result["confidence"], 3),
            "secondary_categories": [{"category": keyword_cat, "confidence": round(keyword_result["confidence"], 3)}],
            "detection_method": "hybrid_ml_win"
        }


def detect_offensive_words(text: str) -> list[str]:
# (function body remains unchanged)
    if not text:
        return []

    # Get cached profanity set (loaded only once)
    profanity_set = _load_profanity_list()

    if not profanity_set:
        # No profanity list available, return empty list
        return []

    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()

    # Split text into words and check against profanity set
    # This handles word boundaries better than substring matching
    words_in_text = set(text_lower.split())
    detected = words_in_text & profanity_set  # Set intersection for O(1) lookup

    # Also check for multi-word phrases in the profanity list
    # (some entries might be phrases like "click here")
    for profane_phrase in profanity_set:
        if ' ' in profane_phrase and profane_phrase in text_lower:
            detected.add(profane_phrase)

    # Convert to sorted list for consistent output
    result = sorted(detected)

    if result:
        print(f"[DEBUG] Detected offensive words: {result}", file=sys.stderr)

    return result


if __name__ == '__main__':
    main()
