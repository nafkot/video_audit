import os
import base64
import requests
import argparse
import json
from PIL import Image
import io
from typing import List, Dict

# --- CONFIGURATION ---
# Ensure these match your local service or update to OpenAI if you move this to cloud API
SERVICE_URL = "http://127.0.0.1:8001"
CLIP_URL = f"{SERVICE_URL}/analyse_frame"
DINO_URL = f"{SERVICE_URL}/find_objects"
DESCRIBE_URL = f"{SERVICE_URL}/describe"
STORAGE_FRAMES = "storage/frames"
OUTPUT_BASE = "output_assets"

API_TIMEOUT = 120

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

BRAND_MAPPING = {
    "adidas": "Adidas", "nike": "Nike", "puma": "Puma", 
    "apple": "Apple", "google": "Google", "samsung": "Samsung",
    "coca-cola": "Coca-Cola", "pepsi": "Pepsi"
    # ... (Add full list as needed)
}

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def describe_scene(image_base64: str) -> str:
    try:
        response = requests.post(DESCRIBE_URL, json={"image_base64": image_base64}, timeout=API_TIMEOUT)
        return response.json().get("description", "").lower()
    except: return ""

def find_objects(image_base64: str) -> List[Dict]:
    try:
        response = requests.post(DINO_URL, json={"image_base64": image_base64, "prompt": "logo"}, timeout=API_TIMEOUT)
        return response.json().get("detections", [])
    except: return []

def analyse_clip(image_base64: str, prompts: List[str]) -> Dict[str, float]:
    try:
        response = requests.post(CLIP_URL, json={"image_base64": image_base64, "prompts": prompts}, timeout=API_TIMEOUT)
        return response.json().get("scores", {})
    except: return {}

def main(video_id: str):
    # Derive frames directory from video_id
    frames_dir = os.path.join(STORAGE_FRAMES, video_id)
    
    if not os.path.exists(frames_dir):
        print(f"❌ Frames directory not found: {frames_dir}")
        exit(1)

    image_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    
    if not image_files:
        print("No frames found to analyze.")
        exit(0)

    aggregated_scores = {cat: {p: 0.0 for p in prompts} for cat, prompts in PROMPT_CATEGORIES.items()}
    identified_brands = {}
    visual_context = []

    print(f"Analysis started: {len(image_files)} frames for {video_id}...")

    # Limit to every 5th frame to speed up if there are too many
    process_files = image_files[::5] if len(image_files) > 20 else image_files

    for i, image_path in enumerate(process_files):
        print(f"Processing frame {i+1}/{len(process_files)}...", end="\r")
        try:
            img = Image.open(image_path)
            img_b64 = encode_image_to_base64(img)
        except: continue

        # 1. Context
        desc = describe_scene(img_b64)
        if desc: visual_context.append(desc)

        # 2. Brand Check
        brand_found_in_text = False
        for keyword, canonical_brand in BRAND_MAPPING.items():
            if keyword in desc:
                identified_brands.setdefault(f"{canonical_brand} logo", []).append(1.0)
                brand_found_in_text = True

        if not brand_found_in_text:
            detections = find_objects(img_b64)
            if detections:
                identified_brands.setdefault("Unknown Logo Object", []).append(detections[0]['score'])

        # 3. Safety
        scores = analyse_clip(img_b64, PROMPT_CATEGORIES["safety"])
        for p, s in scores.items():
            aggregated_scores["safety"][p] += s

    # --- RESULTS ---
    final_report = {
        "visual_context": list(set(visual_context)),
        "brands": {b: len(s) for b, s in identified_brands.items()},
        "safety": {}
    }

    # Calculate Safety Winner
    if process_files:
        avgs = {p: s/len(process_files) for p, s in aggregated_scores["safety"].items()}
        winner = max(avgs, key=avgs.get)
        winner_score = avgs[winner]
        
        final_report["safety"] = {
            "status": "Safe / Low Confidence" if winner_score < 0.50 else winner,
            "score": winner_score
        }

    # Save to JSON
    out_dir = os.path.join(OUTPUT_BASE, video_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{video_id}_visual_analysis.json")
    
    with open(out_path, 'w') as f:
        json.dump(final_report, f, indent=2)

    print(f"\n✅ Visual analysis saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Changed to positional argument to match audit.py call: "python analyse_frames.py [id]"
    parser.add_argument("video_id", help="The ID of the video to analyze")
    args = parser.parse_args()
    main(args.video_id)
