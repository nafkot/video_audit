import os
import time
import requests
import json
import subprocess
from flask import current_app
from pydub import AudioSegment

class AudioService:
    @staticmethod
    def split_stems(audio_path, video_id):
        """
        Uploads audio to LALAL.ai and downloads split stems.
        Returns paths to the saved vocal and instrumental files.
        """
        api_key = current_app.config['LALAL_API_KEY']
        if not api_key:
            return {"error": "LALAL_API_KEY not set"}

        url_upload = "https://www.lalal.ai/api/v1/split/"
        storage_dir = current_app.config['AUDIO_DIR']  # Now saving to storage/audio/ as requested

        # 1. Upload
        try:
            with open(audio_path, 'rb') as f:
                headers = {'Authorization': f'license {api_key}'}
                files = {'file': f}
                data = {'stem': 'vocals', 'filter': 1} # 1 = Mild filter

                print(f"[LALAL] Uploading {os.path.basename(audio_path)}...")
                resp = requests.post(url_upload, headers=headers, files=files, data=data)

            if resp.status_code != 200:
                return {"error": f"LALAL Upload Failed: {resp.text}"}

            task = resp.json()
            task_id = task.get('id')
            if not task_id:
                return {"error": f"Invalid LALAL Response: {task}"}

            # 2. Poll for Completion
            print(f"[LALAL] Processing Task {task_id}...")
            url_check = f"https://www.lalal.ai/api/v1/split/{task_id}"

            while True:
                check = requests.get(url_check, headers=headers).json()
                if check['status'] == 'success':
                    break
                if check['status'] == 'error':
                    return {"error": f"LALAL Processing Error: {check.get('error')}"}
                time.sleep(2)

            # 3. Download Stems
            stem_vocals_url = check['tracks']['vocals']
            stem_instr_url = check['tracks']['accompaniment']

            path_vocals = os.path.join(storage_dir, f"{video_id}_vocals.mp3")
            path_instr = os.path.join(storage_dir, f"{video_id}_instrumentals.mp3")

            AudioService._download_file(stem_vocals_url, path_vocals)
            AudioService._download_file(stem_instr_url, path_instr)

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
        """
        Compares RMS amplitude (loudness) of vocals vs instrumentals.
        Returns a percentage representing how dominant the music is.
        """
        try:
            if not os.path.exists(vocal_path) or not os.path.exists(instr_path):
                return 0.0

            voc = AudioSegment.from_file(vocal_path)
            inst = AudioSegment.from_file(instr_path)

            # dBFS (Decibels relative to Full Scale) is usually negative
            voc_db = voc.dBFS
            inst_db = inst.dBFS

            # Convert to simple linear ratio approximation for the report
            # If music is louder (-10dB) than vocals (-20dB), ratio is high
            # This is a heuristic calculation
            total_energy = abs(voc_db) + abs(inst_db)
            if total_energy == 0: return 0

            # Inverse because lower absolute dB (closer to 0) is louder
            # Example: Voc -20, Inst -10 (Louder). Inst "weight" should be higher.
            # Simple approach: Difference comparison
            diff = inst_db - voc_db # if positive, music is louder

            # Normalize to 0-100% audibility scale (arbitrary scale for audit context)
            # If diff is 0, audibility is 50%. If diff is +10, music dominates.
            audibility = 50 + (diff * 2)
            return max(0, min(100, round(audibility, 2)))

        except Exception as e:
            print(f"Loudness Check Failed: {e}")
            return 0.0
