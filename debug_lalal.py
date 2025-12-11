import os
import sys
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()

# Use the key you provided or load from env
# Note: Ensure this is your 'API Key' from the profile, not just a desktop activation code if they differ.
LALAL_API_KEY = os.getenv("LALAL_API_KEY", "9a88182130da4322")

BASE_URL = "https://www.lalal.ai/api"

def debug_lalal_flow(file_path):
    print(f"--- Debugging LALAL.ai API Flow ---")
    print(f"File: {file_path}")

    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    # Use 'license' or 'key' depending on the specific key type.
    # Standard API usage often uses 'license'.
    headers = {'Authorization': f'license {LALAL_API_KEY}'}

    # --- STEP 1: UPLOAD FILE ---
    print("\n[STEP 1] Uploading file to /v1/input/...")
    upload_url = f"{BASE_URL}/v1/input/"

    file_id = None

    try:
        with open(file_path, 'rb') as f:
            # The API expects the file in the 'file' part of multipart/form-data
            files = {'file': f}
            resp = requests.post(upload_url, headers=headers, files=files)

        print(f"Upload Status: {resp.status_code}")

        if resp.status_code != 200:
            print("❌ Upload Failed")
            print("Response:", resp.text[:500])
            return

        response_json = resp.json()
        file_id = response_json.get('id')
        print(f"✅ File Uploaded! ID: {file_id}")

    except Exception as e:
        print(f"[CRITICAL ERROR] Upload step failed: {e}")
        return

    if not file_id:
        print("❌ No File ID received. Exiting.")
        return

    # --- STEP 2: REQUEST SPLIT ---
    print("\n[STEP 2] Requesting Split (Vocals)...")
    split_url = f"{BASE_URL}/v1/split/"

    # Now we send JSON, not the file itself
    payload = {
        'id': file_id,
        'stem': 'vocals',
        'filter': 1  # Mild filtering
    }

    try:
        # Note: We do NOT send 'files' here, just 'data' or 'json'
        resp = requests.post(split_url, headers=headers, json=payload)

        print(f"Split Request Status: {resp.status_code}")

        if resp.status_code not in [200, 201]:
            print("❌ Split Request Failed")
            print("Response:", resp.text[:500])
            return

        split_json = resp.json()
        print("✅ Split Task Started!")
        print("Response:", json.dumps(split_json, indent=2))

        # Typically returns a task ID or the results immediately if cached/small.
        # If it returns a task ID, you would poll /v1/check/ next.

    except Exception as e:
        print(f"[CRITICAL ERROR] Split step failed: {e}")

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "storage/audio/0kfLmie73dA.mp3"
    debug_lalal_flow(target_file)
