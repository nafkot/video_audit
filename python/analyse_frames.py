import os
import base64
import requests
import argparse
import json
from PIL import Image
import io
from typing import List, Dict, Optional

# --- CONFIGURATION ---
SERVICE_URL = "http://127.0.0.1:8001"
CLIP_URL = f"{SERVICE_URL}/analyse_frame"
DINO_URL = f"{SERVICE_URL}/find_objects"
DESCRIBE_URL = f"{SERVICE_URL}/describe"

API_TIMEOUT = 120 # 2 mins timeout to be safe on CPU

# Improved Safety Prompts to avoid false positives on sketches
PROMPT_CATEGORIES = {
    "safety": [
        "a safe and appropriate image",
        "a normal photograph",
        "a black and white line drawing",
        "a sketch or artwork",
        "a corporate logo or graphic",
        "a digital illustration",
        "a graphic, violent, or gory image",
        "an image containing hateful symbols",
        "an image with nudity or sexual content"
    ]
}

# Brands we trust BLIP to find via text
SIMPLE_BRAND_LIST = [
    "adidas", "nike", "puma", "reebok", "under armour", "gucci",
    "louis vuitton", "chanel", "prada", "apple", "microsoft",
    "google", "amazon", "sony", "samsung", "coca-cola", "pepsi"
]

# MAPPING: Keyword -> Canonical Brand Name
# If BLIP says the keyword (left), we log the Brand (right).
BRAND_MAPPING = {
    "adidas": "Adidas",
    "adi ": "Adidas",        # Catch "adi logo"
    "three stripes": "Adidas",
    "nike": "Nike",
    "swoosh": "Nike",
    "puma": "Puma",
    "reebok": "Reebok",
    "under armour": "Under Armour",
    "gucci": "Gucci",
    "louis vuitton": "Louis Vuitton",
    "lv ": "Louis Vuitton",
    "chanel": "Chanel",
    "prada": "Prada",
    "apple": "Apple",
    "macbook": "Apple",      # Context clues
    "iphone": "Apple",
    "microsoft": "Microsoft",
    "google": "Google",
    "amazon": "Amazon",
    "sony": "Sony",
    "playstation": "Sony",
    "samsung": "Samsung",
    "galaxy": "Samsung",
    "coca-cola": "Coca-Cola",
    "coke": "Coca-Cola",
    "pepsi": "Pepsi"
}
def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def describe_scene(image_base64: str) -> str:
    try:
        response = requests.post(DESCRIBE_URL, json={"image_base64": image_base64}, timeout=API_TIMEOUT)
        return response.json().get("description", "").lower()
    except Exception as e:
        print(f"  [!] Error in description: {e}", flush=True)
        return ""

def find_objects(image_base64: str) -> List[Dict]:
    try:
        # UX: Print status so user knows it's working
        print("  > ...Scanning for small logos...", end="\r", flush=True)
        response = requests.post(DINO_URL, json={"image_base64": image_base64, "prompt": "logo"}, timeout=API_TIMEOUT)
        print(" " * 40, end="\r", flush=True) # Clear line
        return response.json().get("detections", [])
    except: return []

def analyse_clip(image_base64: str, prompts: List[str]) -> Dict[str, float]:
    try:
        response = requests.post(CLIP_URL, json={"image_base64": image_base64, "prompts": prompts}, timeout=API_TIMEOUT)
        return response.json().get("scores", {})
    except: return {}

def main(frames_dir: str):
    image_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    if not image_files: return

    aggregated_scores = {cat: {p: 0.0 for p in prompts} for cat, prompts in PROMPT_CATEGORIES.items()}
    identified_brands = {}
    visual_context = []

    print(f"Analysis started: {len(image_files)} frames.", flush=True)

    for i, image_path in enumerate(image_files):
        print(f"--- Frame {i+1}/{len(image_files)} ---", flush=True)
        try:
            img = Image.open(image_path)
            img_b64 = encode_image_to_base64(img)
        except: continue

        # 1. GET CONTEXT (BLIP)
        desc = describe_scene(img_b64)
        visual_context.append(desc)
        print(f"  > Context: {desc}", flush=True)

        # 2. HYBRID BRAND CHECK (BLIP First -> DINO Second)
        brand_found_in_text = False
        for keyword, canonical_brand in BRAND_MAPPING.items():
            if keyword in desc:
                print(f"  > BLIP Found Brand: {canonical_brand.upper()} (matched '{keyword}')", flush=True)
                identified_brands.setdefault(f"{canonical_brand} logo", []).append(1.0)
                brand_found_in_text = True

        if not brand_found_in_text:
            detections = find_objects(img_b64)
            if detections:
                print(f"  > DINO Found {len(detections)} potential logo(s).", flush=True)
                identified_brands.setdefault("Unknown Logo Object", []).append(detections[0]['score'])
            else:
                print(f"  > No logos detected.", flush=True)

        # 3. SAFETY CHECK
        scores = analyse_clip(img_b64, PROMPT_CATEGORIES["safety"])
        for p, s in scores.items():
            aggregated_scores["safety"][p] += s

    # --- REPORT ---
    final_report = {
        "visual_context": list(set(visual_context)),
        "brands": {b: len(s) for b, s in identified_brands.items()},
        "safety": {}
    }

    # Safety Report with Confidence Threshold
    avgs = {p: s/len(image_files) for p, s in aggregated_scores["safety"].items()}
    winner = max(avgs, key=avgs.get)
    winner_score = avgs[winner]

    if winner_score < 0.50:
        final_report["safety"] = {
            "status": "Safe / Low Confidence",
            "original_guess": winner,
            "score": winner_score
        }
    else:
        final_report["safety"] = {"status": winner, "score": winner_score}

    print("\n" + json.dumps(final_report, indent=2), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id", type=str)
    parser.add_argument("--frames_dir", type=str, required=True)
    main(parser.parse_args().frames_dir)
