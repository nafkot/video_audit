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
API_TIMEOUT = 120
BASE_FRAMES_DIR = "storage/videos/frames/"
MULTIMODAL_MODEL = "gpt-4o" # Recommended for vision and speed
SUMMARY_MODEL = "gpt-3.5-turbo" # Used for long text summary only
POLICY_FILE = "python/policies.json"
MAX_WORKERS = 5 # Maximum concurrent API requests to run

# --- LLM CLIENT INITIALIZATION (Explicit Key Passing) ---
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
        # Reduced error logging here to prevent spamming the console during parallel runs
        return "[LLM_DESC_FAILED]"


def audit_policy_with_llm(image_b64: str, policy_data: List[Dict]) -> List[Dict]:
    """
    Sends the entire policy list AND the image to GPT-4o for direct compliance checking.
    It returns a list of policy audit results or a list containing a single error dictionary.
    """
    if openai_client is None: return [{"Error": "OpenAI client not initialized"}]

    print("\n[+] Multimodal LLM is performing policy audit on first frame...", flush=True)

    # --- PROMPT CONSTRUCTION (Data to be sent) ---
    system_prompt = (
        "You are an expert content policy auditor. Your task is to check the provided image "
        "against the list of policies in the JSON object. You must analyze the image and "
        "determine if each policy in the list is Breached ('yes' or 'no'). "
        "For policies that are Breached: 'yes', provide a brief, specific justification in the 'Violation' field. "
        "For policies that are Breached: 'no', set the 'Violation' field to 'N/A'. "
        "The output MUST be a JSON array containing the original policies with added 'Breached' and 'Violation' fields. "
        "Only include policies that do NOT have the 'Logic_Check' key."
    )

    policy_subset = [p for p in policy_data if 'Logic_Check' not in p]
    policy_prompt_text = json.dumps(policy_subset, indent=2)

    user_prompt_text = (
        f"Analyze this image based on the following content policies. Output the results as a single JSON array.\n\n"
        f"POLICIES TO AUDIT:\n{policy_prompt_text}"
    )

    content_array = [
        {"type": "text", "text": user_prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
    ]
    # --- END PROMPT CONSTRUCTION ---

    start_time = time.time()

    try:
        print("DEBUG: Sending request to OpenAI API...", flush=True)
        response = openai_client.chat.completions.create(
            model=MULTIMODAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_array}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=2048,
            timeout=API_TIMEOUT # Explicit timeout
        )

        print(f"DEBUG: API request finished in {time.time() - start_time:.2f} seconds.", flush=True)

        # --- Post-Response Checks and JSON Decoding ---
        raw_json_text_full = response.choices[0].message.content

        if not raw_json_text_full:
             print("DEBUG: !!! LLM POLICY AUDIT FAILED !!! Error: Received empty or invalid response from API.", flush=True)
             return [{"Error": "LLM Policy Audit Failed: Received empty or invalid response from API."}]

        raw_json_text = raw_json_text_full.strip()

        if not raw_json_text:
             print("DEBUG: !!! LLM POLICY AUDIT FAILED !!! Error: Response content was empty after stripping.", flush=True)
             return [{"Error": "LLM Policy Audit Failed: Response content was empty."}]

        # JSON Decoding with Fallback
        try:
            return json.loads(raw_json_text)
        except json.JSONDecodeError as e:
            # Fallback attempt to repair JSON
            print(f"DEBUG: JSONDecodeError ({e}). Attempting to recover JSON...", flush=True)

            start_index = raw_json_text.find('[')
            end_index = raw_json_text.rfind(']')

            try:
                if start_index != -1 and end_index != -1 and end_index > start_index:
                     repaired_json = raw_json_text[start_index:end_index + 1]
                     return json.loads(repaired_json)
            except json.JSONDecodeError:
                 # If repair fails, execution falls through to the final unconditional failure return
                 pass

        # --- CRITICAL: UNCONDITIONAL FAILURE RETURN ---
        # This block executes if JSON decoding failed, and the fallback also failed.
        print("DEBUG: !!! LLM POLICY AUDIT FAILED !!! Execution flow reached end of try block after JSON decode failure.", flush=True)
        return [{"Error": "LLM Policy Audit Failed: JSON decoding or repair failed unexpectedly."}]

    except Timeout:
        print(f"DEBUG: !!! LLM POLICY AUDIT FAILED !!! Error: API call timed out after {API_TIMEOUT} seconds.", flush=True)
        return [{"Error": f"LLM Policy Audit Failed: API call timed out after {API_TIMEOUT} seconds."}]
    except APIError as e:
        print(f"DEBUG: !!! LLM POLICY AUDIT FAILED !!! Error: OpenAI API Error: {e}", flush=True)
        return [{"Error": f"LLM Policy Audit Failed: OpenAI API Error: {e.status_code} - {e.response.text}"}]
    except Exception as e:
        print(f"DEBUG: !!! LLM POLICY AUDIT FAILED !!! Error: Unexpected failure: {e}", flush=True)
        return [{"Error": f"LLM Policy Audit Failed: Unexpected failure: {e}"}]


def get_llm_final_summary(detailed_story: str) -> str:
    """Sends the detailed narrative to the OpenAI API for single-sentence synthesis."""
    if openai_client is None: return "[LLM_API_FAILED]"

    print("\n[+] Generating FINAL STORYLINE via GPT-3.5...", flush=True)

    try:
        prompt_content = (
            f"Condense the following detailed visual narrative sequence into one high-level sentence, focusing on the main actions and overall transition: {detailed_story}"
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
        # Note: We silence the error message to avoid flooding the console during parallel runs
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


def audit_narrative(narrative_text: str, static_ratio: float, first_frame_b64: str, policy_list: List[Dict]) -> List[Dict]:
    """Primary auditing function. Uses LLM for the main policy check (first frame)
    and handles the static ratio (logic check)."""

    # 1. LLM Multimodal Policy Check (for first, most critical frame)
    llm_audit_results = audit_policy_with_llm(first_frame_b64, policy_list)

    # 2. Manual Static Ratio Check (Logic Check)
    static_ratio_violation = {
        "Category": "Content ID & CMS Policies",
        "Policy": "Content ID Circumvention (Static Visuals)",
        "Description": "Claiming short loops, movie clips, or static image videos just to farm revenue. Breached if static frame ratio > 15%.",
        "Breached": "yes" if static_ratio > 15.0 else "no",
        "Violation": f"Static ratio {static_ratio:.1f}%"
    }

    # Handle LLM failure or append logic check

    # 1. If LLM call failed and returned None or an empty list, handle it here.
    if not llm_audit_results:
        # This is a highly defensive check; the audit_policy_with_llm should prevent an empty list return.
        return [{"Error": "LLM Policy Audit Failed: The audit function returned an empty or invalid result."}]

    # 2. If the LLM call returned an explicit error object (e.g., due to API failure/timeout), return it.
    if "Error" in llm_audit_results[0]:
        return llm_audit_results

    # Find and update the Static Visuals policy in the LLM results, or append it if missing.
    found_static = False
    for i, item in enumerate(llm_audit_results):
        if item.get("Policy") == "Content ID Circumvention (Static Visuals)":
            llm_audit_results[i] = static_ratio_violation
            found_static = True
            break

    if not found_static:
        llm_audit_results.append(static_ratio_violation)

    return llm_audit_results


def process_frame(file_path: str):
    """Encodes a single frame and gets the LLM description."""
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

    # --- FIXED: Parallel Processing with Guaranteed Order ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 1. Submit tasks and store futures with their original index
        future_to_index = {
            executor.submit(process_frame, file_path): i
            for i, file_path in enumerate(files)
        }

        # 2. Collect results in the correct sequence using the original index
        # We use as_completed but then use the index to correctly place the result in all_results
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]

            # Log completion status (optional, but good for UX)
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

    for i, result in enumerate(all_results):
        desc, b64 = result
        if not desc or desc == "[LLM_DESC_FAILED]":
            continue

        if i == 0:
            first_frame_b64 = b64 # Capture the first frame's B64 for policy audit

        if desc != last_desc:
            descriptions.append(desc)
            last_desc = desc
        else:
            repeated_count += 1

    # --- Generation and Analysis ---
    detailed_narrative, final_summary_line = construct_storyline(descriptions)
    full_narrative_text = " ".join(descriptions)
    static_ratio = (repeated_count / len(files)) * 100

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
