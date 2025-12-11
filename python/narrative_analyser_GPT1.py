import os
import argparse
import json
import base64
import io
import sys
import concurrent.futures
from PIL import Image
from typing import List, Dict, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
API_TIMEOUT = 30
# Ensure this matches where your splitter puts frames
BASE_FRAMES_DIR = "storage/videos/frames/" 
OUTPUT_ASSETS_DIR = "output_assets"
MULTIMODAL_MODEL = "gpt-4o"
SUMMARY_MODEL = "gpt-3.5-turbo"
POLICY_FILE = "python/policies.json"
MAX_WORKERS = 5
POLICY_MAX_WORKERS = 2

# --- LLM CLIENT INITIALIZATION ---
API_KEY_VALUE = os.getenv("OPENAI_API_KEY")

try:
    if not API_KEY_VALUE:
        raise ValueError("OPENAI_API_KEY environment variable not found.")
    openai_client = OpenAI(api_key=API_KEY_VALUE)
    print(f"[+] Multimodal Model Client Initialized...", flush=True)
except Exception as e:
    print("FATAL ERROR: Failed to initialize OpenAI client.", flush=True)
    openai_client = None

# --- UTILITIES ---

def load_policies(filename: str) -> List[Dict]:
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"CRITICAL ERROR loading policies: {e}", file=sys.stderr)
        return []

def load_transcript(video_id: str) -> str:
    transcript_path = os.path.join(OUTPUT_ASSETS_DIR, video_id, f"{video_id}_transcription.json")
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, 'r') as f:
                data = json.load(f)
                return data.get('text') or data.get('transcript') or data.get('full_text') or ""
        except:
            pass
    return ""

def save_report(video_id: str, report_data: Dict):
    """Saves the final report to the path expected by the Flask app."""
    output_dir = os.path.join(OUTPUT_ASSETS_DIR, video_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # The Flask app expects: {video_id}_narrative_report.json
    filename = f"{video_id}_narrative_report.json"
    output_path = os.path.join(output_dir, filename)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"[+] Report saved successfully to: {output_path}")
    except Exception as e:
        print(f"[!] Failed to save report: {e}")

def encode_image_to_base64(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB': img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding {image_path}: {e}")
        return None

# --- LLM FUNCTIONS ---

def get_llm_description(image_b64: str, prompt: str) -> str:
    if not openai_client: return "[LLM_DESC_FAILED]"
    try:
        response = openai_client.chat.completions.create(
            model=MULTIMODAL_MODEL,
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except:
        return "[LLM_DESC_FAILED]"

def audit_single_policy_with_llm(image_b64: str, policy_data: Dict, full_context_text: str) -> Dict:
    if not openai_client: return {"Policy": policy_data['Policy'], "Error": "No Client"}

    system_prompt = (
        "You are an expert content policy auditor. Check the video content against the policy. "
        "Use the provided Image and the Text Summary (Narrative + Audio). "
        "If 'Inappropriate Language' or 'Hate Speech', prioritize Audio. "
        "Output a single JSON object with keys: 'Category', 'Policy', 'Description', 'Breached' (yes/no), 'Violation' (reason/NA)."
    )
    
    user_prompt = (
        f"POLICY: {json.dumps(policy_data)}\n\n"
        f"CONTEXT: {full_context_text}\n\n"
        f"Return JSON."
    )

    try:
        response = openai_client.chat.completions.create(
            model=MULTIMODAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"Policy": policy_data['Policy'], "Breached": "error", "Violation": str(e)}

def construct_storyline(descriptions: List[str], transcript: str) -> Tuple[str, str]:
    if not openai_client: return "LLM Error", "LLM Error"
    
    narrative_text = " ".join(descriptions)
    
    try:
        response = openai_client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": "Summarize this video narrative and transcript into one cohesive sentence."},
                {"role": "user", "content": f"Visuals: {narrative_text}\nAudio: {transcript}"}
            ]
        )
        summary = response.choices[0].message.content.strip()
        return narrative_text, summary
    except:
        return narrative_text, "Summary generation failed."

def main(video_id: str):
    print(f"--- Starting GPT-1 Audit for {video_id} ---", flush=True)
    
    # 1. Load Data
    policies = load_policies(POLICY_FILE)
    transcript = load_transcript(video_id)
    frames_dir = os.path.join(BASE_FRAMES_DIR, video_id)
    
    # Fallback path check
    if not os.path.exists(frames_dir):
        frames_dir = f"storage/frames/{video_id}"
    
    if not os.path.exists(frames_dir):
        print(f"[!] Frames dir not found: {frames_dir}")
        return

    files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not files:
        print("[!] No frames found.")
        return

    # 2. Analyze Frames (Parallel)
    print(f"[+] analyzing {len(files)} frames...", flush=True)
    descriptions = []
    first_frame_b64 = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exc:
        futures = {exc.submit(process_frame, f): i for i, f in enumerate(files)}
        results = [None] * len(files)
        for fut in concurrent.futures.as_completed(futures):
            results[futures[fut]] = fut.result()
            
    # Filter results
    valid_results = [r for r in results if r and r[0] != "[LLM_DESC_FAILED]"]
    descriptions = [r[0] for r in valid_results]
    if valid_results:
        first_frame_b64 = valid_results[0][1]

    # 3. Generate Narrative & Audit
    narrative, summary = construct_storyline(descriptions, transcript)
    
    context_text = f"Visuals: {narrative}\nAudio: {transcript}"
    
    print("[+] Running Policy Audit...", flush=True)
    audit_results = []
    
    # Filter for LLM policies only
    llm_policies = [p for p in policies if 'Logic_Check' not in p]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=POLICY_MAX_WORKERS) as exc:
        futures = [exc.submit(audit_single_policy_with_llm, first_frame_b64, p, context_text) for p in llm_policies]
        for fut in concurrent.futures.as_completed(futures):
            audit_results.append(fut.result())

    # 4. Construct Final Report Structure
    final_report = {
        "narrative": narrative,
        "summary": summary,
        "policy_analysis": audit_results,
        "transcript_used": bool(transcript)
    }

    # 5. Save to File (Critical for Flask)
    save_report(video_id, final_report)
    print("--- Audit Complete ---", flush=True)

def process_frame(path):
    b64 = encode_image_to_base64(path)
    if not b64: return None
    return get_llm_description(b64, "Describe the main action."), b64

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    args = parser.parse_args()
    main(args.video_id)
