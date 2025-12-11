import os
import time
import requests
import json
from pydub import AudioSegment

class AudioService:
    @staticmethod
    def split_stems(audio_path, video_id, api_key, storage_dir, log_callback=None):
        """
        Uploads audio to LALAL.ai and downloads split stems.
        """
        def log(msg):
            if log_callback: log_callback(msg)
            else: print(msg)

        if not api_key:
            return {"error": "LALAL_API_KEY not provided"}

        # Use v1 for stability, check documentation if v2 is required
        url_upload = "https://www.lalal.ai/api/v1/split/"

        try:
            # 1. Upload
            log(f"[LALAL] Uploading {os.path.basename(audio_path)} ({os.path.getsize(audio_path)} bytes)...")

            with open(audio_path, 'rb') as f:
                headers = {'Authorization': f'license {api_key}'}
                files = {'file': f}
                data = {'stem': 'vocals', 'filter': 1}

                resp = requests.post(url_upload, headers=headers, files=files, data=data)

            if resp.status_code != 200:
                log(f"[LALAL] Error: HTTP {resp.status_code}")
                return {"error": f"LALAL Upload Failed: {resp.text[:200]}"}

            try:
                task = resp.json()
            except:
                return {"error": "LALAL returned non-JSON response"}

            task_id = task.get('id')
            if not task_id:
                return {"error": f"Invalid LALAL Response: {task}"}

            log(f"[LALAL] Task ID: {task_id}. Processing...")

            # 2. Poll for Completion
            url_check = f"https://www.lalal.ai/api/v1/split/{task_id}"

            while True:
                check = requests.get(url_check, headers=headers).json()
                status = check.get('status')

                if status == 'success':
                    break
                if status == 'error':
                    return {"error": f"LALAL Processing Error: {check.get('error')}"}

                log(f"[LALAL] Status: {status}...")
                time.sleep(3) # Wait 3s before retry

            # 3. Download Stems
            log("[LALAL] Downloading stems...")
            stem_vocals_url = check['tracks']['vocals']
            stem_instr_url = check['tracks']['accompaniment']

            os.makedirs(storage_dir, exist_ok=True)

            path_vocals = os.path.join(storage_dir, f"{video_id}_vocals.mp3")
            path_instr = os.path.join(storage_dir, f"{video_id}_instrumentals.mp3")

            AudioService._download_file(stem_vocals_url, path_vocals)
            AudioService._download_file(stem_instr_url, path_instr)

            log("[LALAL] Download complete.")

            return {
                "success": True,
                "vocals_path": path_vocals,
                "instr_path": path_instr
            }

        except Exception as e:
            return {"error": f"LALAL Exception: {str(e)}"}

    @staticmethod
    def _download_file(url, save_path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    @staticmethod
    def calculate_audibility(vocal_path, instr_path):
        try:
            if not os.path.exists(vocal_path) or not os.path.exists(instr_path):
                return 0.0
            voc = AudioSegment.from_file(vocal_path)
            inst = AudioSegment.from_file(instr_path)
            diff = inst.dBFS - voc.dBFS
            audibility = 50 + (diff * 2)
            return max(0, min(100, round(audibility, 2)))
        except: return 0.0
