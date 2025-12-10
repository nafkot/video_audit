import os
import requests
import argparse
import json
import base64
import io
import time
from PIL import Image
from typing import List, Dict, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
SERVICE_URL = "http://127.0.0.1:8001/describe"
API_TIMEOUT = 120
BASE_FRAMES_DIR = "storage/videos/frames/" 

# --- LLM CLIENT INITIALIZATION (Explicit Key Passing) ---
API_KEY_VALUE = os.getenv("OPENAI_API_KEY")

try:
    if not API_KEY_VALUE:
        raise ValueError("OPENAI_API_KEY environment variable not found.")
    openai_client = OpenAI(api_key=API_KEY_VALUE)
    print(f"[+] OpenAI Client Initialized. Key starts with: {API_KEY_VALUE[:5]}...", flush=True)

except Exception as e:
    print("FATAL ERROR: Failed to initialize OpenAI client.", flush=True)
    print(f"Error details: {e}", flush=True)
    openai_client = None

# --- KEYWORD LISTS ---
PROFANITY_KEYWORDS = ["fuck", "shit", "bitch", "cunt", "asshole"]
VIOLENCE_KEYWORDS = ["blood", "gore", "severe injury", "violent acts", "gunfire", "screaming", "gun", "knife", "wrestler"] 
SEXUAL_KEYWORDS = ["naked", "sex", "nude", "explicit", "genitalia", "fetish", "suggestive", "undressed", "strip"]
HARMFUL_KEYWORDS = ["drug use", "self-harm", "eating disorder", "challenge", "drug paraphernalia"]
SCAM_KEYWORDS = ["scam", "phishing", "malware", "clickbait", "misleading", "fraudulent"]
CONTROVERSIAL_KEYWORDS = ["armed conflicts", "recent tragedies", "divisive political events"]
FIREARMS_KEYWORDS = ["firearm", "weapon", "sell", "manufacture", "modify"]
SPONSORSHIP_KEYWORDS = ["sponsor", "partner", "sponsored", "promotion", "advertisement"] 
HATE_KEYWORDS = ["hate", "racism", "supremacy", "bigoted", "slurs", "derogatory terms"]


# --- POLICY DEFINITION (Unchanged) ---
POLICY_AUDIT_LIST = [
    { "Category": "Community Guidelines Violations", "Policy": "Hate Speech", "Description": "Content promoting violence or hatred against individuals or groups based on protected attributes (race, religion, gender, etc.).", "Keywords": HATE_KEYWORDS },
    { "Category": "Community Guidelines Violations", "Policy": "Harassment & Cyberbullying", "Description": "Maliciously insulting, threatening, or doxxing (revealing personal info) an individual.", "Keywords": ["insulting", "threatening", "doxxing"] },
    { "Category": "Community Guidelines Violations", "Policy": "Violent or Graphic Content", "Description": "Showing shocking, gory, or violent acts (e.g., severe injuries, active violence) for shock value, not in an educational or news context.", "Keywords": VIOLENCE_KEYWORDS },
    { "Category": "Community Guidelines Violations", "Policy": "Nudity & Sexual Content", "Description": "Content depicting sexual acts, genitalia, or fetish content.", "Keywords": SEXUAL_KEYWORDS },
    { "Category": "Community Guidelines Violations", "Policy": "Child Safety", "Description": "Any content that endangers minors, including sexualization, harmful acts, or bullying.", "Keywords": ["minor", "child", "kid", "suggestive", "danger"] },
    { "Category": "Community Guidelines Violations", "Policy": "Harmful or Dangerous Acts", "Description": "Showing or promoting dangerous 'challenges,' drug use, self-harm, eating disorders, or instructional content for harmful activities.", "Keywords": HARMFUL_KEYWORDS },
    { "Category": "Community Guidelines Violations", "Policy": "Spam, Deceptive Practices & Scams", "Description": "Misleading thumbnails/titles ('clickbait'), keyword stuffing, or links to external phishing/malware/scam sites.", "Keywords": SCAM_KEYWORDS },
    { "Category": "Advertiser-Friendly Content Guidelines", "Policy": "Inappropriate Language", "Description": "Frequent or severe profanity (e.g., F-words, C-words, slurs), especially in the first 30 seconds of a video.", "Keywords": PROFANITY_KEYWORDS },
    { "Category": "Advertiser-Friendly Content Guidelines", "Policy": "Adult Themes", "Description": "Content that is 'sexually suggestive' but not explicit, such as crude humor or innuendo-heavy discussions.", "Keywords": ["crude humor", "innuendo", "suggestive"] },
    { "Category": "Advertiser-Friendly Content Guidelines", "Policy": "Controversial Subjects", "Description": "Discussing highly sensitive, recent, and divisive topics (e.g., armed conflicts, recent tragedies, divisive political events) in a way that is not neutral news reporting.", "Keywords": CONTROVERSIAL_KEYWORDS },
    { "Category": "Advertiser-Friendly Content Guidelines", "Policy": "Firearms-Related Content", "Description": "Content focused on the sale, manufacture, or modification of firearms, or showing them in a harmful way.", "Keywords": FIREARMS_KEYWORDS },
    { "Category": "Content ID & CMS Policies", "Policy": "Content ID Circumvention (Static Visuals)", "Description": "Claiming short loops, movie clips, or static image videos just to farm revenue. Breached if static frame ratio > 15%.", "Logic_Check": "STATIC_RATIO" },
    { "Category": "Content ID & CMS Policies", "Policy": "Content ID Circumvention (Audio Inaudibility)", "Description": "Abusing the Content ID system with claims on invalid content. Flagged if claimed audio asset is at -30dB or lower while the main track is loud.", "Logic_Check": "PLACEHOLDER_AUDIO" },
    { "Category": "Content ID & CMS Policies", "Policy": "Artificial/Fraudulent Claims", "Description": "Claiming content that is not owned, such as public domain music, royalty-free loops, white noise, or AI-generated soundscapes.", "Keywords": ["public domain", "royalty-free", "white noise", "AI-generated soundscapes"] },
    { "Category": "Content ID & CMS Policies", "Policy": "Third-Party Copyright", "Description": "A creator in your network using popular music, movie clips, or sports footage that you don't own.", "Keywords": ["popular music", "movie clips", "sports footage"] },
    { "Category": "Content ID & CMS Policies", "Policy": "Sponsorship Disclosure", "Description": "Creators who have a paid promotion but fail to check the 'Includes paid promotion' box.", "Keywords": SPONSORSHIP_KEYWORDS }
]


# --- SERVICE CALLS & UTILITIES (Defined correctly) ---

def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image file to a base64 string for API calls."""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding {image_path}: {e}", flush=True)
        return None

def get_blip_description(image_b64: str) -> str:
    """Asks the BLIP service to describe the image content."""
    try:
        response = requests.post("http://127.0.0.1:8001/describe", json={"image_base64": image_b64}, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json().get("description", "").strip()
    except Exception as e:
        # Added debug logging to catch connection issues
        print(f"DEBUG: Error calling BLIP service: {e}", flush=True)
        return ""

def get_llm_final_summary(detailed_story: str) -> str:
    """
    Sends the detailed narrative to the OpenAI API for single-sentence synthesis.
    """
    if openai_client is None:
        return "LLM Summary Failed: OpenAI client not initialized."
        
    print("\n[+] Generating FINAL STORYLINE via GPT-3.5...", flush=True)

    try:
        prompt_content = (
            f"Condense the following detailed visual narrative sequence into one high-level sentence, focusing on the main actions and overall transition: {detailed_story}"
        )
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a concise video analysis assistant. Your task is to condense a sequence of visual events into a single, flowing, cohesive summary sentence."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.2, 
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"DEBUG: !!! LLM API CALL FAILED !!! Error: {e}", flush=True)
        # --- FALLBACK FIX: Provide an informative message instead of a generic string ---
        return f"LLM Summary Failed: API Error. Check key/internet."


def construct_storyline(unique_descriptions: List[str]) -> Tuple[str, str]:
    """
    Generates the detailed, transition-based narrative (rule-based) and
    then calls the LLM for the final summary line.
    """
    if not unique_descriptions: 
        return "No discernible visual narrative.", "LLM Summary Failed: No content to summarize."

    detailed_story_lines = []
    detailed_story_lines.append(f"The video opens with {unique_descriptions[0]}.")
    
    for i in range(1, len(unique_descriptions)):
        prev_desc = unique_descriptions[i-1].lower()
        curr_desc = unique_descriptions[i]
        
        # Heuristics for transition phrases
        if any(keyword in curr_desc.lower() for keyword in ["shifts to", "focuses on", "appears", "standing", "setting"]) or len(unique_descriptions) > 5:
            transition = "Following this action, the scene shifts to"
        elif any(subj in prev_desc for subj in ["person", "man", "woman", "wrestler"]):
            transition = "They continue, and next, we see"
        else:
            transition = "After a brief moment, we transition to"
            
        detailed_story_lines.append(f"{transition} {curr_desc}.")

    detailed_story_text = "\n".join(detailed_story_lines)

    # Call the LLM to get the final summary sentence
    final_summary_line = get_llm_final_summary(detailed_story_text)

    # --- LOCAL FALLBACK CHECK (Only for display if LLM failed) ---
    if "LLM Summary Failed" in final_summary_line:
        # If the LLM failed, create a simple, non-repetitive summary from the local data.
        first_desc = unique_descriptions[0].split('a ', 1)[-1].split('the ', 1)[-1]
        last_desc = unique_descriptions[-1].split('a ', 1)[-1].split('the ', 1)[-1]
        
        # Combine the failure message with the local summary for completeness
        fallback_summary = f"[LLM ERROR: Using Local Fallback] The visual narrative transitions from {first_desc} to {last_desc}."
        return detailed_story_text, fallback_summary


    return detailed_story_text, final_summary_line


def audit_narrative(narrative_text: str, static_ratio: float) -> List[Dict]:
    """Checks the narrative against the predefined policies and returns a JSON-serializable list."""
    narrative_lower = narrative_text.lower()
    results = []

    for policy_def in POLICY_AUDIT_LIST:
        breached = "no"
        violation_details = "none"
        
        if "Keywords" in policy_def:
            found_words = [k for k in policy_def["Keywords"] if k in narrative_lower]
            if policy_def["Policy"] == "Severe Profanity":
                profanity_roots = ["fuck", "shit", "cunt"]
                found_words = [w for w in profanity_roots if w in narrative_lower]

            if found_words:
                breached = "yes"
                violation_details = list(set(found_words))

        elif policy_def.get("Logic_Check") == "STATIC_RATIO":
            violation_details = f"Static ratio {static_ratio:.1f}%"
            if static_ratio > 15.0:
                breached = "yes"
        
        elif policy_def.get("Logic_Check") == "PLACEHOLDER_AUDIO":
            violation_details = "Requires Audio/dB Analysis"
            
        results.append({
            "Category": policy_def["Category"],
            "Policy": policy_def["Policy"],
            "Description": policy_def["Description"],
            "Breached": breached,
            "Violation": violation_details
        })
        
    return results

def main(video_id: str):
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
    
    for i, file_path in enumerate(files):
        print(f"Processing frame {i+1}/{len(files)}...", end="\r", flush=True)
        b64 = encode_image_to_base64(file_path)
        if not b64: continue
        desc = get_blip_description(b64)
        
        if desc:
            if desc == last_desc:
                repeated_count += 1
            if desc != last_desc:
                descriptions.append(desc)
                last_desc = desc
    
    print(" " * 50, end="\r") 

    # --- Generation and Analysis ---
    detailed_narrative, final_summary_line = construct_storyline(descriptions)
    full_narrative_text = " ".join(descriptions)
    static_ratio = (repeated_count / len(files)) * 100
    
    policy_report_json = audit_narrative(full_narrative_text, static_ratio)

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
