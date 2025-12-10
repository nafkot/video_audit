import os
import requests
import argparse
import json
import base64
import io
import time
import sys
import concurrent.futures
from PIL import Image
from typing import List, Dict, Tuple, Any
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# --- CONFIGURATION ---
API_TIMEOUT = 30
BASE_FRAMES_DIR = "storage/frames"  # Fixed path
MULTIMODAL_MODEL = "gpt-4o-mini"
SUMMARY_MODEL = "gpt-4o-mini"
POLICY_FILE = "python/policies.json"
POLICY_MAX_WORKERS = 1
MAX_WORKERS = 5
# Define output directory relative to the project root
OUTPUT_ASSETS_DIR = "output_assets"

# --- LLM CLIENT INITIALIZATION ---
API_KEY_VALUE = os.getenv("OPENAI_API_KEY")
try:
    if not API_KEY_VALUE: raise ValueError("OPENAI_API_KEY not found.")
    openai_client = OpenAI(api_key=API_KEY_VALUE)
except Exception as e:
    print(f"FATAL ERROR: {e}", flush=True)
    openai_client = None

def load_policies(filename: str) -> List[Dict]:
    """Loads policies and ensures default keys exist."""
    try:
        with open(filename, 'r') as f:
            policies = json.load(f)
            # Initialize required keys to prevent KeyError later
            for p in policies:
                p.setdefault('Breached', 'check_required')
                p.setdefault('Violation', 'N/A')
            return policies
    except Exception as e:
        print(f"CRITICAL ERROR loading policies: {e}", file=sys.stderr)
        sys.exit(1)

def encode_image_to_base64(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB': img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding {image_path}: {e}", flush=True)
        return None

def get_llm_description(image_b64: str, prompt: str) -> str:
    if openai_client is None: return "[LLM_DESC_FAILED]"
    try:
        response = openai_client.chat.completions.create(
            model=MULTIMODAL_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]}],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception: return "[LLM_DESC_FAILED]"

def process_frame(file_path: str):
    b64 = encode_image_to_base64(file_path)
    if not b64: return None, None
    desc = get_llm_description(b64, "Describe this scene concisely.")
    return desc, b64

def audit_single_policy_with_llm(image_b64: str, policy_data: Dict) -> Dict:
    if openai_client is None: return {"Policy": policy_data['Policy'], "Breached": "error", "Violation": "No API"}

    for attempt in range(3):
        try:
            system_prompt = "You are a content policy auditor. Check the image against the policy. Return JSON: {\"Policy\": \"...\", \"Breached\": \"yes/no\", \"Violation\": \"reason\"}"
            policy_json = json.dumps(policy_data)
            response = openai_client.chat.completions.create(
                model=MULTIMODAL_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Audit this policy:\n{policy_json}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            time.sleep(2)
    return {"Policy": policy_data['Policy'], "Breached": "error", "Violation": "LLM Failed"}

def construct_storyline(descriptions: List[str]) -> Tuple[str, str]:
    if not descriptions: return "No content.", "No content."
    text = " ".join(descriptions)
    try:
        resp = openai_client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[{"role": "user", "content": f"Summarize this video narrative in one sentence: {text[:4000]}"}]
        )
        return text, resp.choices[0].message.content.strip()
    except: return text, "Summary generation failed."

def audit_narrative(static_ratio: float, first_frame_b64: str, policy_list: List[Dict]) -> Dict:
    # 1. Filter Visual Policies
    visual_policies = [p for p in policy_list if "Audio" not in p.get('Category', '')]

    # 2. Run Audit
    results = []
    print(f"Auditing {len(visual_policies)} policies...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(audit_single_policy_with_llm, first_frame_b64, p): p for p in visual_policies}
        for f in concurrent.futures.as_completed(futures):
            try: results.append(f.result())
            except: results.append({"Policy": futures[f]['Policy'], "Breached": "error"})

    # 3. Merge Results
    final_policies = []
    # Add Visual Results
    for p in policy_list:
        p_copy = p.copy()
        # Find result if it exists
        res = next((r for r in results if r.get('Policy') == p['Policy']), None)
        if res:
            p_copy['Breached'] = res.get('Breached', 'error')
            p_copy['Violation'] = res.get('Violation', 'N/A')
        else:
            # Mark skipped/audio policies as N/A
            p_copy['Breached'] = 'N/A'
            p_copy['Violation'] = 'Skipped (Audio/Text analysis disabled)'
        final_policies.append(p_copy)

    # 4. Calculate Score
    breaches = sum(1 for p in final_policies if p['Breached'] == 'yes')
    total = len(final_policies)
    score = int(100 - (breaches / total * 100)) if total > 0 else 0

    return {
        "overall_score": score,
        "policy_checks": final_policies,
        "static_ratio": static_ratio
    }

def main(video_id: str):
    print(f"--- Starting Narrative Analysis for {video_id} ---", flush=True)

    # 1. Setup Paths
    frames_dir = os.path.join(BASE_FRAMES_DIR, video_id)
    out_dir = os.path.join(OUTPUT_ASSETS_DIR, video_id)
    os.makedirs(out_dir, exist_ok=True)

    # 2. Load Images
    if not os.path.exists(frames_dir):
        print(f"Error: Frames dir not found: {frames_dir}", flush=True)
        return
    files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    if not files:
        print("No frames found.", flush=True)
        return

    # 3. Analyze Frames (Sample first 5 for speed/cost in demo)
    print("Analyzing frames...", flush=True)
    # Get first frame for policy check
    _, first_b64 = process_frame(files[0])

    # 4. Audit
    policies = load_policies(POLICY_FILE)
    report = audit_narrative(0.0, first_b64, policies)

    # 5. SAVE REPORT TO FILE (Critical Fix)
    out_path = os.path.join(out_dir, f"{video_id}_narrative_report.json")
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✅ Report saved to: {out_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id", type=str)
    args = parser.parse_args()
    main(args.video_id)
