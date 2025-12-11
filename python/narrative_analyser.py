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

load_dotenv()

# --- CONFIGURATION ---
API_TIMEOUT = 30
BASE_FRAMES_DIR = "storage/videos/frames"
MULTIMODAL_MODEL = "gpt-4o-mini"
SUMMARY_MODEL = "gpt-4o-mini"
POLICY_FILE = "python/policies.json"
POLICY_MAX_WORKERS = 1
MAX_WORKERS = 5
OUTPUT_ASSETS_DIR = "output_assets"

API_KEY_VALUE = os.getenv("OPENAI_API_KEY")
try:
    if not API_KEY_VALUE: raise ValueError("OPENAI_API_KEY not found.")
    openai_client = OpenAI(api_key=API_KEY_VALUE)
except Exception as e:
    print(f"FATAL ERROR: {e}", flush=True)
    openai_client = None

def load_policies(filename: str) -> List[Dict]:
    try:
        with open(filename, 'r') as f:
            policies = json.load(f)
            for p in policies:
                p.setdefault('Breached', 'check_required')
                p.setdefault('Violation', 'N/A')
            return policies
    except Exception as e:
        print(f"CRITICAL ERROR loading policies: {e}", file=sys.stderr)
        return []

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
    except Exception as e: 
        return f"[LLM_DESC_FAILED: {str(e)}]"

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

def run_concurrent_policy_audit(audit_func, audit_source, policies):
    """Checks policies one by one and PRINTS progress immediately."""
    print(f"\n[DEBUG] Starting audit of {len(policies)} policies...", flush=True)
    results = []
    
    # Sequential Loop for Clear Debugging
    for i, policy in enumerate(policies):
        p_name = policy.get('Policy', 'Unknown')
        print(f"[DEBUG] [{i+1}/{len(policies)}] Checking: {p_name}...", end=" ", flush=True)
        
        try:
            res = audit_func(audit_source, policy)
            status = res.get('Breached', 'error')
            results.append(res)
            print(f"-> {status.upper()}", flush=True)
        except Exception as e:
            print(f"-> ERROR: {e}", flush=True)
            results.append({"Policy": p_name, "Breached": "error", "Violation": str(e)})
            
    return results

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
    
    # 2. Run Audit (With Debug Logging)
    results = run_concurrent_policy_audit(audit_single_policy_with_llm, first_frame_b64, visual_policies)

    # 3. Merge Results
    final_policies = []
    for p in policy_list:
        p_copy = p.copy()
        res = next((r for r in results if r.get('Policy') == p['Policy']), None)
        if res:
            p_copy['Breached'] = res.get('Breached', 'error')
            p_copy['Violation'] = res.get('Violation', 'N/A')
        else:
            p_copy['Breached'] = 'N/A'
            p_copy['Violation'] = 'Skipped (Audio/Text analysis disabled)'
        final_policies.append(p_copy)

    # 4. Score
    # Fix: Count "yes" AND "Yes" as breaches
    breaches = sum(1 for p in final_policies if str(p['Breached']).lower() == 'yes')
    total = len(final_policies)
    score = int(100 - (breaches / total * 100)) if total > 0 else 0

    return {
        "overall_score": score,
        "policy_checks": final_policies,
        "static_ratio": static_ratio
    }

def main(video_id: str):
    print(f"--- Starting Narrative Analysis for {video_id} ---", flush=True)
    
    frames_dir = os.path.join(BASE_FRAMES_DIR, video_id)
    out_dir = os.path.join(OUTPUT_ASSETS_DIR, video_id)
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(frames_dir):
        print(f"[ERROR] Frames dir not found: {frames_dir}", flush=True)
        return
        
    files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    if not files:
        print("[ERROR] No frames found.", flush=True)
        return

    # Analyze Frames with Progress
    print(f"[DEBUG] Found {len(files)} frames. Analyzing samples...", flush=True)
    descriptions = []
    first_b64 = None
    
    # Sample every 5th frame to save time/cost, or process all if needed
    sample_files = files[::5] 
    
    for i, f_path in enumerate(sample_files):
        print(f"[DEBUG] Analyzing Frame {i+1}/{len(sample_files)}...", end=" ", flush=True)
        desc, b64 = process_frame(f_path)
        
        if b64 and not first_b64: first_b64 = b64
        if desc: descriptions.append(desc)
        print("Done.", flush=True)
    
    # Summary
    print("[DEBUG] Generating Visual Summary...", flush=True)
    detailed, summary = construct_storyline(descriptions)
    
    # Policy Audit
    policies = load_policies(POLICY_FILE)
    if first_b64:
        report = audit_narrative(0.0, first_b64, policies)
        report['visual_summary'] = summary  # <--- SAVE SUMMARY FOR REPORT.HTML
    else:
        report = {"overall_score": 0, "error": "No valid frames analyzed"}

    # Save
    out_path = os.path.join(out_dir, f"{video_id}_narrative_report.json")
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Report saved to: {out_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id", type=str)
    args = parser.parse_args()
    main(args.video_id)
