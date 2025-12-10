import os
import requests
import argparse
import json
import base64
import io
import time
import sys
import concurrent.futures
import subprocess
from PIL import Image
from typing import List, Dict, Tuple, Any
from openai import OpenAI, APIError, Timeout, RateLimitError
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# --- CONFIGURATION ---
API_TIMEOUT = 30 
BASE_FRAMES_DIR = "storage/videos/frames/"
MULTIMODAL_MODEL = "gpt-4o-mini"
SUMMARY_MODEL = "gpt-4o-mini"
POLICY_FILE = "python/policies.json"
POLICY_MAX_WORKERS = 1 # Force sequential policy check
MAX_WORKERS = 5 
AUDIO_ANALYSIS_SCRIPT = Path(__file__).parent / 'analyze.py'
VIDEO_AUDIO_WAV = 'storage/videos/audio.wav'
AUDIO_ANALYSIS_OUTPUT_PATH = "storage/reports/{video_id}_audio_analysis.json" 

# --- LLM CLIENT INITIALIZATION ---
API_KEY_VALUE = os.getenv("OPENAI_API_KEY")

try:
    if not API_KEY_VALUE:
        raise ValueError("OPENAI_API_KEY environment variable not found.")
    openai_client = OpenAI(api_key=API_KEY_VALUE)
    print(f"[+] OpenAI Client Initialized. Using Multimodal Model: {MULTIMODAL_MODEL}", flush=True)

except Exception as e:
    print("FATAL ERROR: Failed to initialize OpenAI client.", flush=True)
    print(f"Error details: {e}", flush=True)
    openai_client = None

# --- Function to Load Policies (Unchanged) ---
def load_policies(filename: str) -> List[Dict]:
    """Loads and parses the policy list from the external JSON file."""
    try:
        with open(filename, 'r') as f:
            policies = json.load(f)
            if not all('Category' in p and 'Policy' in p for p in policies):
                raise ValueError("Policy file is missing essential fields.")
            return policies
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Policy file '{filename}' not found. Exiting.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"CRITICAL ERROR: Policy file '{filename}' is invalid JSON. Check formatting.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR loading policies: {e}", file=sys.stderr)
        sys.exit(1)


# --- SERVICE CALLS & UTILITIES ---

def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image file to a base64 string for API calls."""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding {image_path}: {e}", flush=True)
        return None

def get_llm_description(image_b64: str, prompt: str) -> str:
    """Sends image to LLM for detailed description."""
    if openai_client is None: return "[LLM_DESC_FAILED]"

    try:
        content_array = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]

        response = openai_client.chat.completions.create(
            model=MULTIMODAL_MODEL,
            messages=[{"role": "user", "content": content_array}],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return "[LLM_DESC_FAILED]"


def process_frame(file_path: str):
    """Encodes a single frame and gets the LLM description."""
    b64 = encode_image_to_base64(file_path)
    if not b64: return None, None

    prompt = "Describe this scene concisely, focusing on the main subject and action."
    desc = get_llm_description(b64, prompt)

    return desc, b64


def audit_single_policy_text(audit_text: str, policy_data: Dict) -> Dict:
    """
    Sends a single policy AND the text (English transcript) to LLM for compliance checking.
    (Function body remains unchanged, used only if audio audit is enabled)
    """
    if openai_client is None: return {"Policy": policy_data['Policy'], "Breached": "error", "Violation": "OpenAI client not initialized"}

    # --- RETRY CONFIGURATION ---
    max_retries = 5
    retry_delay = 5  # Start with 5 seconds

    for attempt in range(max_retries):
        try:
            # --- PROMPT CONSTRUCTION ---
            system_prompt = (
                "You are an expert content policy auditor. Your task is to check the provided text "
                "against the single policy in the JSON object. "
                "Determine if the policy is Breached ('yes' or 'no'). "
                "**CRITICAL INSTRUCTION**: If the policy is 'Inappropriate Language', check for profanity in the text. "
                "If Breached: 'yes', provide a brief, specific justification in the 'Violation' field. "
                "If Breached: 'no', set the 'Violation' field to 'N/A'. "
                "The output MUST be a single JSON object containing ONLY the policy name, 'Breached', and 'Violation' fields. "
                "Output ONLY the JSON object, with no other text."
            )

            policy_prompt_text = json.dumps(policy_data, indent=2)

            user_prompt_text = (
                f"Analyze the following audio transcript based on the policy provided. Output a single JSON object with the policy name, 'Breached', and 'Violation' status.\n\n"
                f"TRANSCRIPT: \"{audit_text}\"\n\n"
                f"POLICY TO AUDIT:\n{policy_prompt_text}"
            )

            response = openai_client.chat.completions.create(
                model=SUMMARY_MODEL, # Use the faster text model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=256,
                timeout=API_TIMEOUT
            )

            raw_json_text = response.choices[0].message.content.strip()

            if not raw_json_text:
                raise ValueError("Received empty response from API.")

            # JSON Decoding and Repair (re-using logic from previous fixes)
            try:
                results = json.loads(raw_json_text)
                if isinstance(results, dict):
                    return results
                else:
                    raise json.JSONDecodeError("Parsed JSON was not a single object.", raw_json_text, 0)
            except json.JSONDecodeError:
                start_index = raw_json_text.find('{')
                end_index = raw_json_text.rfind('}')
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    repaired_json = raw_json_text[start_index:end_index + 1]
                    try:
                        results = json.loads(repaired_json)
                        if isinstance(results, dict): return results
                    except json.JSONDecodeError: pass

                raise ValueError(f"JSON decoding or repair failed.")

        except RateLimitError as e:
            if attempt < max_retries - 1:
                print(f"\n[RATE LIMIT] Hit 429. Retrying policy {policy_data['Policy']} in {retry_delay}s (Attempt {attempt + 1}/{max_retries})...", file=sys.stderr, flush=True)
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                # Max retries reached
                return {
                    "Policy": policy_data['Policy'],
                    "Breached": "error",
                    "Violation": f"LLM Audit Failed: RateLimitError after {max_retries} retries."
                }

        except Exception as e:
            # Handle other errors (Timeout, APIError, JSONDecodeError)
            return {
                "Policy": policy_data['Policy'],
                "Breached": "error",
                "Violation": f"LLM Audit Failed: {type(e).__name__} - {str(e)}"
            }

def audit_single_policy_with_llm(image_b64: str, policy_data: Dict) -> Dict:
    """
    Sends a single policy AND the image to LLM for compliance checking (Visual Audit).
    """
    if openai_client is None: return {"Policy": policy_data['Policy'], "Breached": "error", "Violation": "OpenAI client not initialized"}

    # --- RETRY CONFIGURATION ---
    max_retries = 5
    retry_delay = 5  # Start with 5 seconds

    for attempt in range(max_retries):
        try:
            # --- PROMPT CONSTRUCTION ---
            system_prompt = (
                "You are an expert content policy auditor. Your task is to check the provided image "
                "against the single policy in the JSON object. "
                "Determine if the policy is Breached ('yes' or 'no'). "
                "**CRITICAL INSTRUCTION**: If the policy is 'Violent or Graphic Content', and the image shows real or simulated severe injuries, explicit gore, blood, or objects embedded in a person's body (like skewers), you MUST classify the policy as Breached: 'yes'. "
                "If Breached: 'yes', provide a brief, specific justification in the 'Violation' field. "
                "If Breached: 'no', set the 'Violation' field to 'N/A'. "
                "The output MUST be a single JSON object containing ONLY the policy name, 'Breached', and 'Violation' fields. "
                "Output ONLY the JSON object, with no other text."
            )

            policy_prompt_text = json.dumps(policy_data, indent=2)

            user_prompt_text = (
                f"Analyze this image based on the following content policy. Output a single JSON object with the policy name, 'Breached', and 'Violation' status.\n\n"
                f"POLICY TO AUDIT:\n{policy_prompt_text}"
            )

            content_array = [
                {"type": "text", "text": user_prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
            # --- END PROMPT CONSTRUCTION ---

            response = openai_client.chat.completions.create(
                model=MULTIMODAL_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_array}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=256,
                timeout=API_TIMEOUT
            )

            raw_json_text = response.choices[0].message.content.strip()

            if not raw_json_text:
                raise ValueError("Received empty response from API.")

            # JSON Decoding and Repair (re-using logic from previous fixes)
            try:
                results = json.loads(raw_json_text)
                if isinstance(results, dict):
                    return results
                else:
                    raise json.JSONDecodeError("Parsed JSON was not a single object.", raw_json_text, 0)
            except json.JSONDecodeError:
                start_index = raw_json_text.find('{')
                end_index = raw_json_text.rfind('}')
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    repaired_json = raw_json_text[start_index:end_index + 1]
                    try:
                        results = json.loads(repaired_json)
                        if isinstance(results, dict):
                            return results
                    except json.JSONDecodeError:
                        pass
                raise ValueError(f"JSON decoding or repair failed. Raw response: {raw_json_text[:100]}...")


        except RateLimitError as e:
            if attempt < max_retries - 1:
                print(f"\n[RATE LIMIT] Hit 429. Retrying policy {policy_data['Policy']} in {retry_delay}s (Attempt {attempt + 1}/{max_retries})...", file=sys.stderr, flush=True)
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                # Max retries reached
                return {
                    "Policy": policy_data['Policy'],
                    "Breached": "error",
                    "Violation": f"LLM Audit Failed: RateLimitError after {max_retries} retries."
                }

        except Exception as e:
            # Handle other errors (Timeout, APIError, JSONDecodeError)
            return {
                "Policy": policy_data['Policy'],
                "Breached": "error",
                "Violation": f"LLM Audit Failed: {type(e).__name__} - {str(e)}"
            }

def get_llm_final_summary(detailed_story: str) -> str:
    """Sends the detailed narrative to the LLM API for single-sentence synthesis."""
    if openai_client is None: return "[LLM_API_FAILED]"

    print("\n[+] Generating FINAL STORYLINE via LLM...", flush=True)

    try:
        prompt_content = (
            f"Condense the following detailed visual narrative sequence into one high-level sentence. "
            f"The summary **must** focus on the sequence of events, including the **most extreme acts of violence** (e.g., strikes, use of weapons) and the resulting **visible injuries** (e.g., blood, objects in the head). "
            f"Detailed Narrative: {detailed_story}"
        )

        response = openai_client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise video analysis assistant. Your task is to condense a sequence of visual events into a single, flowing, cohesive summary sentence."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.2,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return "[LLM_API_FAILED]"


def construct_storyline(unique_descriptions: List[str]) -> Tuple[str, str]:
    # --- Detail Generation (Rule-based) ---
    if not unique_descriptions:
        return "No discernible visual narrative.", "LLM Summary Failed: No content to summarize."

    detailed_story_lines = []
    detailed_story_lines.append(f"The video opens with {unique_descriptions[0]}.")

    for i in range(1, len(unique_descriptions)):
        prev_desc = unique_descriptions[i-1].lower()
        curr_desc = unique_descriptions[i]

        if any(keyword in curr_desc.lower() for keyword in ["shifts to", "focuses on", "appears", "standing", "setting"]) or len(unique_descriptions) > 5:
            transition = "Following this action, the scene shifts to"
        elif any(subj in prev_desc for subj in ["person", "man", "woman", "wrestler"]):
            transition = "They continue, and next, we see"
        else:
            transition = "After a brief moment, we transition to"

        detailed_story_lines.append(f"{transition} {curr_desc}.")

    detailed_story_text = "\n".join(detailed_story_lines)

    # --- Final Summary (LLM Call or Fallback) ---
    final_summary_line = get_llm_final_summary(detailed_story_text)

    if final_summary_line.startswith("[LLM_API_FAILED]"):
        first_desc = unique_descriptions[0].split('a ', 1)[-1].split('the ', 1)[-1].strip()
        last_desc = unique_descriptions[-1].split('a ', 1)[-1].split('the ', 1)[-1].strip()

        # Fallback summary (non-repetitive)
        fallback_summary = f"[API OFFLINE: Using Local Summary] The visual narrative begins with a scene of {first_desc}, transitions through the video's main events, and concludes with a shot of {last_desc}."
        return detailed_story_text, fallback_summary

    return detailed_story_text, final_summary_line


def run_concurrent_policy_audit(audit_func: callable, audit_source: Any, policies: List[Dict]) -> List[Dict]:
    """
    Orchestrates the policy audit by running individual policy checks in parallel.
    Uses audit_func (either visual or text) and audit_source (B64 or transcript).
    """
    if not policies:
        return []

    print(f"\n[+] Starting concurrent LLM audit of {len(policies)} policies using {audit_func.__name__}...", flush=True)

    results = []

    # FIX: Use POLICY_MAX_WORKERS (which is 1) to eliminate the concurrent burst
    with concurrent.futures.ThreadPoolExecutor(max_workers=POLICY_MAX_WORKERS) as executor:
        future_to_policy = {
            executor.submit(audit_func, audit_source, policy): policy
            for policy in policies
        }

        for i, future in enumerate(concurrent.futures.as_completed(future_to_policy)):
            try:
                result = future.result()
                results.append(result)
                print(f"Policy audit progress: {i+1}/{len(policies)} policies checked.", end='\r', flush=True)
            except Exception as exc:
                policy = future_to_policy[future]
                error_result = {
                    "Policy": policy['Policy'],
                    "Breached": "error",
                    "Violation": f"Execution Error during concurrency: {exc}"
                }
                results.append(error_result)

    print(" " * 60, end='\r')
    print(f"[+] Concurrent policy audit finished. ({len(results)} results collected).", flush=True)
    return results


def process_audio_analysis(video_id: str):
    """
    (DISABLED) Runs the external audio analysis script and loads the resulting JSON file.
    """
    # This function is now bypassed by returning empty data.
    print("[INFO] Audio analysis pipeline disabled.", file=sys.stderr, flush=True)
    return {
        "error": "Audio analysis disabled by user.",
        "ai_generated": False, "ai_confidence": 0,
        "detected_low_volume": False, "low_volume_percent": 0,
        "found": False, "similarity": 0,
        "vocals_analysis": {"language": "N/A", "transcript": "N/A", "transcript_english": "", "sentiment": {}, "topic": {}, "offensive_content": False, "offensive_words": []}
    }


def audit_narrative(full_narrative_text: str, static_ratio: float, first_frame_b64: str, policy_list: List[Dict], audio_analysis: Dict) -> Dict:
    """
    Primary auditing function. Integrates visual, audio, LLM, and logic checks.
    """

    # --- 0. Split Policies into Visual and Audio Categories ---
    # Since audio analysis is disabled, we merge all audio-specific policies into a disabled list
    visual_policies_to_run = [p for p in policy_list if p['Policy'] not in [
        "Inappropriate Language",
        "Adult Themes",
        "Controversial Subjects",
        "Content ID Circumvention (Audio Inaudibility)",
        "Artificial/Fraudulent Claims"
    ]]
    audio_policies_disabled = [p for p in policy_list if p['Policy'] not in visual_policies_to_run]

    # --- 1. Run Concurrent LLM Policy Checks ---

    # a. Visual Audit (Multimodal)
    visual_llm_results = run_concurrent_policy_audit(
        audit_single_policy_with_llm,
        first_frame_b64,
        visual_policies_to_run
    )

    # b. Audio Audit (DISABLED)
    audio_llm_results = []
    
    # --- 2. Build Report Structures ---

    # Helper to populate base structure for both reports
    def get_base_report_data(source_type: str, analysis_data: Dict):
        vocals_data = analysis_data.get('vocals_analysis', {})
        topic_data = vocals_data.get('topic', {})
        sentiment_data = vocals_data.get('sentiment', {})

        # Populate metadata needed for the report
        metadata = {
            "language": vocals_data.get('language', 'N/A'),
            "transcript": vocals_data.get('transcript', 'N/A'),
            "sentiment": {
                "label": sentiment_data.get('label', 'N/A'),
                "polarity": sentiment_data.get('polarity', 0),
                "subjectivity": sentiment_data.get('subjectivity', 0)
            },
            "topic": {
                "category": topic_data.get('category', 'N/A'),
                "confidence": topic_data.get('confidence', 0),
                "secondary_categories": topic_data.get('secondary_categories', [])
            }
        }
        
        # Add source-specific technical data (minimal for disabled audio)
        if source_type == "audio":
            metadata.update({
                "ai_generated": analysis_data.get('ai_generated', False),
                "ai_confidence": analysis_data.get('ai_confidence', 0),
                "detected_low_volume": analysis_data.get('detected_low_volume', False),
                "low_volume_percent": analysis_data.get('low_volume_percent', 0),
                "error_reason": analysis_data.get('error', 'N/A')
            })
        elif source_type == "visual":
            metadata.update({
                "static_ratio": static_ratio,
            })
            
        return metadata

    # Initialize final report structure (using placeholder data for audio section)
    final_report = {
        "VISUAL_POLICY_ANALYSIS": get_base_report_data("visual", audio_analysis),
        "AUDIO_POLICY_ANALYSIS": get_base_report_data("audio", audio_analysis)
    }

    # Helper to merge LLM results into the policy list structure
    def merge_llm_results(llm_results: List[Dict], base_policies: List[Dict], disabled_reason: str = None):
        policy_map = {p['Policy']: dict(p) for p in base_policies}
        
        for result in llm_results:
            policy_name = result.get('Policy')
            if policy_name in policy_map:
                policy_map[policy_name].update({
                    'Breached': result.get('Breached', 'error'),
                    'Violation': result.get('Violation', 'LLM result missing violation field')
                })
        
        # Mark disabled policies
        if disabled_reason:
            for policy in policy_map.values():
                if policy['Breached'] == 'check_required' or policy['Breached'] == 'error':
                     policy.update({
                        'Breached': 'N/A',
                        'Violation': disabled_reason
                    })

        return list(policy_map.values())

    # --- 3. Finalize Visual Policies (Merge LLM and Logic Checks) ---
    
    visual_report_policies = merge_llm_results(visual_llm_results, visual_policies_to_run)

    # Manual Logic Check: Static Visuals
    static_policy_name = "Content ID Circumvention (Static Visuals)"
    static_policy = next((p for p in visual_report_policies if p['Policy'] == static_policy_name), None)
    if static_policy:
        is_breached = "yes" if static_ratio > 15.0 else "no"
        static_policy.update({
            'Breached': is_breached,
            'Violation': f"Static ratio {static_ratio:.1f}%"
        })

    final_report["VISUAL_POLICY_ANALYSIS"]["policies"] = visual_report_policies


    # --- 4. Finalize Audio Policies (Mark as Disabled) ---
    disabled_reason = "Audio analysis pipeline disabled by user request."
    
    # Merge LLM results (empty) with ALL audio policies and mark as disabled
    audio_report_policies = merge_llm_results(audio_llm_results, audio_policies_disabled, disabled_reason=disabled_reason)
    
    final_report["AUDIO_POLICY_ANALYSIS"]["policies"] = audio_report_policies

    return final_report


def main(video_id: str):
    # Load policies dynamically at the start
    policy_list = load_policies(POLICY_FILE)

    frames_dir = os.path.join(BASE_FRAMES_DIR, video_id)
    # (Rest of frame loading and description logic remains similar)
    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory not found: {frames_dir}")
        return

    files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    if not files:
        print(f"No frames found in {frames_dir}.")
        return

    print(f"--- Analyzing Narrative for ID: {video_id} ({len(files)} frames) ---", flush=True)

    descriptions = []
    last_desc = ""
    repeated_count = 0
    first_frame_b64 = None
    all_results = [None] * len(files) # Array to store results in submission order

    # --- Parallel Processing for Frame Descriptions ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(process_frame, file_path): i
            for i, file_path in enumerate(files)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            print(f"Frame {index+1}/{len(files)} processed...", end='\r', flush=True)

            try:
                desc, b64 = future.result()
                all_results[index] = (desc, b64)
            except Exception as exc:
                print(f'\nFrame {index+1} generated an exception: {exc}')
                all_results[index] = (None, None) 

    print(" " * 60, end='\r')

    # 3. Process collected results sequentially
    print("Collecting and filtering descriptions sequentially...", flush=True)
    total_processed_frames = 0

    for i, result in enumerate(all_results):
        desc, b64 = result
        if b64:
            total_processed_frames += 1

        if not desc or desc == "[LLM_DESC_FAILED]":
            continue

        if i == 0:
            first_frame_b64 = b64 # Capture the first frame's B64 for policy audit

        if desc != last_desc:
            descriptions.append(desc)
            last_desc = desc
        else:
            repeated_count += 1

    if total_processed_frames == 0 or len(files) == 0:
        static_ratio = 100.0
    else:
        static_ratio = (repeated_count / len(files)) * 100

    # --- Generation and Analysis ---
    detailed_narrative, final_summary_line = construct_storyline(descriptions)
    full_narrative_text = " ".join(descriptions)

    # --- NEW: Audio Analysis Execution (BYPASSED) ---
    audio_analysis_data = process_audio_analysis(video_id) # Call the disabled function
    
    # --- FINAL MITIGATION: FORCED DELAY (Still relevant after summary call) ---
    # Pause execution to allow OpenAI TPM quota to reset after the Narrative Summary call
    print("[WAIT] Applying 5-second buffer delay to reset OpenAI TPM quota...", flush=True)
    time.sleep(5)
    # -----------------------------------

    # Run combined audit
    if first_frame_b64: # Removed 'error' check since the disabled function returns placeholder data
        policy_report_json = audit_narrative(full_narrative_text, static_ratio, first_frame_b64, policy_list, audio_analysis_data)
    else:
        policy_report_json = {"CRITICAL_ERROR": "Failed to run combined audit due to missing frame data."}


    # --- FINAL TEXT OUTPUT ---
    print("\n" + "="*60)
    print(f"VIDEO ID: {video_id}")
    print("NARRATIVE SUMMARY (Full Progression):")
    print("-" * 20)
    print(detailed_narrative)
    print("\nFINAL STORYLINE:")
    print(final_summary_line)
    print("="*60 + "\n")

    print("POLICY ANALYSIS (JSON Output):")
    print(json.dumps(policy_report_json, indent=2))
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate narrative and check policies from video frames.")
    parser.add_argument("video_id", type=str, help="The YouTube video ID (e.g., veZMkE9OaZ8)")
    args = parser.parse_args()
    main(args.video_id)
