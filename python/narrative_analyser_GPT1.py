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
from typing import List, Dict, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from openai import OpenAI, APIError, Timeout

load_dotenv()

# --- CONFIGURATION ---
API_TIMEOUT = 30 # Reduced timeout since each call is small
BASE_FRAMES_DIR = "storage/videos/frames/"
MULTIMODAL_MODEL = "gpt-4o"
SUMMARY_MODEL = "gpt-3.5-turbo"
POLICY_FILE = "python/policies.json"
MAX_WORKERS = 5 # Maximum concurrent API requests to run
POLICY_MAX_WORKERS = 2 # Maximum concurrent API requests to run

# --- LLM CLIENT INITIALIZATION (Explicit Key Passing) ---
API_KEY_VALUE = os.getenv("OPENAI_API_KEY")

try:
    if not API_KEY_VALUE:
        raise ValueError("OPENAI_API_KEY environment variable not found.")
    openai_client = OpenAI(api_key=API_KEY_VALUE)
    print(f"[+] Multimodal Model Client Initialized...", flush=True)

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
    """Sends image to GPT-4o for detailed description."""
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


def audit_single_policy_with_llm(image_b64: str, policy_data: Dict) -> Dict:
    """
    Sends a single policy AND the image to GPT-4o for compliance checking.
    It returns a single policy dictionary with 'Breached' and 'Violation' fields added.
    """
    if openai_client is None: return {"Policy": policy_data['Policy'], "Error": "OpenAI client not initialized"}

    # --- PROMPT CONSTRUCTION ---
    system_prompt = (
        "You are an expert content policy auditor. Your task is to check the provided image "
        "against the single policy in the JSON object. "
        "Determine if the policy is Breached ('yes' or 'no'). "
        "**CRITICAL INSTRUCTION**: If the policy is 'Inappropriate Language', check for profanity in the text. "
        "If Breached: 'yes', provide a brief, specific justification in the 'Violation' field. "
        "If Breached: 'no', set the 'Violation' field to 'N/A'. "
        "The output MUST be a single JSON object containing the original policy fields plus 'Breached' and 'Violation'. "
        "Output ONLY the JSON object, with no other text."
    )

    policy_prompt_text = json.dumps(policy_data, indent=2)

    user_prompt_text = (
        f"Analyze this image based on the following content policy. Output the results as a single JSON object.\n\n"
        f"POLICY TO AUDIT:\n{policy_prompt_text}"
    )

    content_array = [
        {"type": "text", "text": user_prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
    ]
    # --- END PROMPT CONSTRUCTION ---

    start_time = time.time()

    try:
        # print(f"DEBUG: Auditing policy: {policy_data['Policy']}...", flush=True)
        response = openai_client.chat.completions.create(
            model=MULTIMODAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_array}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1024,
            timeout=API_TIMEOUT
        )

        raw_json_text = response.choices[0].message.content.strip()

        if not raw_json_text:
             raise ValueError("Received empty response from API.")

        # Direct JSON Decoding
        try:
            results = json.loads(raw_json_text)
            if isinstance(results, dict):
                return results
            else:
                 raise json.JSONDecodeError("Parsed JSON was not a single object.", raw_json_text, 0)

        except json.JSONDecodeError:
            # Fallback attempt to repair JSON (Optimized for missing outer brackets)
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


    except Exception as e:
        # Generic error handling for API issues, timeouts, or decoding failures
        return {
            "Category": policy_data.get('Category', 'N/A'),
            "Policy": policy_data['Policy'],
            "Breached": "error",
            "Violation": f"LLM Audit Failed: {type(e).__name__} - {str(e)}"
        }

def get_llm_final_summary(detailed_story: str) -> str:
    """Sends the detailed narrative to the OpenAI API for single-sentence synthesis."""
    if openai_client is None: return "[LLM_API_FAILED]"

    print("\n[+] Generating FINAL STORYLINE ...", flush=True)

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

def run_concurrent_policy_audit(first_frame_b64: str, policy_list: List[Dict]) -> List[Dict]:
    """
    Orchestrates the policy audit by running individual policy checks in parallel.
    """
    if not first_frame_b64:
        return [{"Error": "CRITICAL: First frame data is missing."}]

    # Filter out logic checks to be handled later
    llm_policies = [p for p in policy_list if 'Logic_Check' not in p]

    print(f"\n[+] Starting concurrent LLM audit of {len(llm_policies)} policies...", flush=True)

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=POLICY_MAX_WORKERS) as executor:
        # Submit all policy checks
        future_to_policy = {
            executor.submit(audit_single_policy_with_llm, first_frame_b64, policy): policy
            for policy in llm_policies
        }

        # Collect results
        for i, future in enumerate(concurrent.futures.as_completed(future_to_policy)):
            try:
                result = future.result()
                results.append(result)
                print(f"Policy audit progress: {i+1}/{len(llm_policies)} policies checked.", end='\r', flush=True)
            except Exception as exc:
                policy = future_to_policy[future]
                error_result = {
                    "Policy": policy['Policy'],
                    "Breached": "error",
                    "Violation": f"Execution Error during concurrency: {exc}"
                }
                results.append(error_result)

    print(" " * 60, end='\r') # Clear progress line
    print("[+] Concurrent policy audit finished.", flush=True)
    return results


def audit_narrative(narrative_text: str, static_ratio: float, first_frame_b64: str, policy_list: List[Dict]) -> List[Dict]:
    """
    Primary auditing function. Runs concurrent LLM checks and merges results
    with manual logic checks.
    """

    # 1. Run Concurrent LLM Policy Checks
    llm_audit_results = run_concurrent_policy_audit(first_frame_b64, policy_list)

    # If the setup failed (missing B64), return error immediately
    if "Error" in llm_audit_results[0]:
        return llm_audit_results

    # 2. Prepare the final report map with all expected policies
    final_report_map = {
        p['Policy']: {
            'Category': p['Category'],
            'Policy': p['Policy'],
            'Description': p['Description'],
            'Breached': 'check_required',
            'Violation': 'Not checked by LLM or Logic Check'
        } for p in policy_list
    }

    # 3. Incorporate LLM Results
    for result in llm_audit_results:
        policy_name = result.get('Policy')

        # Ensure we only process valid dictionary results with a Policy name
        if isinstance(result, dict) and policy_name and policy_name in final_report_map:
            # Update the map with the LLM's Breached and Violation status
            final_report_map[policy_name].update({
                'Breached': result.get('Breached', 'error'),
                'Violation': result.get('Violation', 'LLM result missing violation field')
            })
        elif policy_name:
            print(f"WARNING: LLM audit for policy '{policy_name}' failed or returned unusable data.", flush=True)

    # 4. Manual Static Ratio Check (Logic Check: STATIC_RATIO)
    static_policy_name = "Content ID Circumvention (Static Visuals)"
    if static_policy_name in final_report_map:
        is_breached = "yes" if static_ratio > 15.0 else "no"
        final_report_map[static_policy_name].update({
            'Breached': is_breached,
            'Violation': f"Static ratio {static_ratio:.1f}%"
        })

    # 5. Manual Audio Check (Logic Check: PLACEHOLDER_AUDIO)
    audio_policy_name = "Content ID Circumvention (Audio Inaudibility)"
    if audio_policy_name in final_report_map:
        final_report_map[audio_policy_name].update({
            'Breached': 'no',
            'Violation': 'Logic check skipped: Requires audio data analysis.'
        })

    # 6. Convert map back to a list for the final output
    return list(final_report_map.values())


def process_frame(file_path: str):
    """Encodes a single frame and gets the LLM description."""
    # (function body remains unchanged)
    b64 = encode_image_to_base64(file_path)
    if not b64: return None, None

    prompt = "Describe this scene concisely, focusing on the main subject and action."
    desc = get_llm_description(b64, prompt)

    return desc, b64


def main(video_id: str):
    # Load policies dynamically at the start
    policy_list = load_policies(POLICY_FILE)

    frames_dir = os.path.join(BASE_FRAMES_DIR, video_id)

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

    # --- Parallel Processing with Guaranteed Order ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(process_frame, file_path): i
            for i, file_path in enumerate(files)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            print(f"Frame {index+1}/{len(files)} processed...", end="\r", flush=True)

            try:
                desc, b64 = future.result()
                all_results[index] = (desc, b64)
            except Exception as exc:
                print(f'\nFrame {index+1} generated an exception: {exc}')
                all_results[index] = (None, None) # Mark as failed

    print(" " * 60, end="\r") # Clear progress line

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

    # Ensure static ratio is calculated against frames that successfully yielded a description
    if total_processed_frames == 0 or len(files) == 0:
        static_ratio = 100.0
    else:
        static_ratio = (repeated_count / len(files)) * 100 # Use total files for ratio

    # --- Generation and Analysis ---
    detailed_narrative, final_summary_line = construct_storyline(descriptions)
    full_narrative_text = " ".join(descriptions)

    # Only run audit if the first frame B64 was successfully captured
    policy_report_json = audit_narrative(full_narrative_text, static_ratio, first_frame_b64, policy_list)

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
