from flask import Blueprint, render_template, request, Response, stream_with_context, current_app, send_file
import json
import time
import os
import subprocess
import glob
from datetime import datetime
from app.services.media_service import MediaService
from app.services.audio_service import AudioService

audit_bp = Blueprint('audit', __name__)

def save_log(video_id, message):
    """Appends a log message to a file."""
    try:
        log_dir = os.path.join(current_app.config['STORAGE_DIR'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"{video_id}.log")
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} {message}\n")
    except Exception as e:
        print(f"Logging failed: {e}")

def run_script(command, video_id):
    """Runs a shell command and yields output for the UI while saving to file."""
    try:
        save_log(video_id, f"EXEC: {command}")
        
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        
        for line in iter(process.stdout.readline, ''):
            clean_line = line.strip()
            if clean_line:
                save_log(video_id, clean_line) # <--- SAVE TO FILE
                yield clean_line
        
        process.stdout.close()
        process.wait()
        
        if process.returncode != 0:
            err_msg = f"[ERROR] Script failed with code {process.returncode}"
            save_log(video_id, err_msg)
            yield err_msg
            
    except Exception as e:
        err_msg = f"[ERROR] Execution failed: {e}"
        save_log(video_id, err_msg)
        yield err_msg

@audit_bp.route('/report/<video_id>')
def view_report(video_id):
    base_storage = current_app.config['STORAGE_DIR']
    output_dir = os.path.join(current_app.config['BASE_DIR'], 'output_assets', video_id)

    # Initialize Data
    data = {
        "meta": {},
        "report": {},
        "transcript": [],
        "music": {},
        "ai_audio": {}
    }

    # Helper to load JSON
    def load_json(filename):
        path = os.path.join(output_dir, f"{video_id}_{filename}")
        if os.path.exists(path):
            with open(path, 'r') as f: return json.load(f)
        return {}

    data['meta'] = load_json("details.json")
    data['report'] = load_json("narrative_report.json")
    data['transcript'] = load_json("transcription.json")
    data['music'] = load_json("music.json")
    data['ai_audio'] = load_json("ai_audio.json")

    # Media
    media = {
        "vocal": f"/media/storage/stems/{video_id}_vocal.mp3",
        "instr": f"/media/storage/stems/{video_id}_no_vocals.mp3"
    }

    # Frames
    frames_dir = os.path.join(base_storage, "frames", video_id)
    frames = []
    if os.path.exists(frames_dir):
        frames = [f"/media/storage/frames/{video_id}/{f}" for f in sorted(os.listdir(frames_dir)) if f.endswith('.jpg')]

    return render_template('report.html', video_id=video_id, data=data, media=media, frames=frames)

@audit_bp.route('/audit/logs/<video_id>')
def download_logs(video_id):
    """Route to view/download the raw log file."""
    log_dir = os.path.join(current_app.config['STORAGE_DIR'], 'logs')
    log_file = f"{video_id}.log"
    return send_file(os.path.join(log_dir, log_file), as_attachment=False, mimetype='text/plain')

@audit_bp.route('/audit/stream/<video_id>')
def stream_audit(video_id):
    def generate():
        # Clear previous log
        log_dir = os.path.join(current_app.config['STORAGE_DIR'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        open(os.path.join(log_dir, f"{video_id}.log"), 'w').close()
        
        save_log(video_id, "--- STARTING AUDIT ---")
        yield f"data: {json.dumps({'message': '🚀 Starting Audit Pipeline...'})}\n\n"
        
        try:
            # 1. Metadata
            yield f"data: {json.dumps({'step': 0, 'message': 'Fetching Metadata...'})}\n\n"
            for log in run_script(f"python3 python/get_youtube_video_metadata.py --id='{video_id}'", video_id):
                yield f"data: {json.dumps({'message': log})}\n\n"

            # 2. Download
            yield f"data: {json.dumps({'step': 1, 'message': 'Downloading Video...'})}\n\n"
            for log in run_script(f"python3 python/get_video_file.py --id='{video_id}'", video_id):
                yield f"data: {json.dumps({'message': log})}\n\n"

            # 3. Frames
            yield f"data: {json.dumps({'step': 2, 'message': 'Splitting Frames...'})}\n\n"
            for log in run_script(f"python3 python/split_video_into_frames.py --fps 0.33 --files '{video_id}.mp4'", video_id):
                yield f"data: {json.dumps({'message': log})}\n\n"

            # 4. Audio Extract
            yield f"data: {json.dumps({'step': 3, 'message': 'Extracting Audio...'})}\n\n"
            vid_path = os.path.join("storage", "videos", f"{video_id}.mp4")
            aud_path = os.path.join("storage", "audio", f"{video_id}.mp3")
            
            subprocess.run(f"ffmpeg -i {vid_path} -q:a 0 -map a {aud_path} -y", shell=True)
            save_log(video_id, "Audio extracted via ffmpeg.")
            yield f"data: {json.dumps({'message': 'Audio extracted.'})}\n\n"

            # 5. LALAL.AI
            yield f"data: {json.dumps({'step': 4, 'message': 'Separating Vocals (LALAL.ai)...'})}\n\n"
            stems = AudioService.split_stems(aud_path, video_id)
            if stems.get("error"):
                 msg = f"[Error] {stems['error']}"
                 save_log(video_id, msg)
                 yield f"data: {json.dumps({'message': msg})}\n\n"
            else:
                 msg = "Stems separated successfully."
                 save_log(video_id, msg)
                 yield f"data: {json.dumps({'message': msg})}\n\n"

            # 6. Music ID
            yield f"data: {json.dumps({'step': 5, 'message': 'Checking Music...'})}\n\n"
            for log in run_script(f"python3 python/detect_music.py {video_id}", video_id):
                yield f"data: {json.dumps({'message': log})}\n\n"

            # 7. AI Audio Check (NEW)
            yield f"data: {json.dumps({'step': 6, 'message': 'Checking AI Voice...'})}\n\n"
            for log in run_script(f"python3 python/detect_ai_audio.py {video_id}", video_id):
                yield f"data: {json.dumps({'message': log})}\n\n"

            # 8. Final Report
            yield f"data: {json.dumps({'step': 7, 'message': 'Generating Final Policy Report...'})}\n\n"
            for log in run_script(f"python3 python/narrative_analyser.py {video_id}", video_id):
                yield f"data: {json.dumps({'message': log})}\n\n"

            save_log(video_id, "--- AUDIT COMPLETE ---")
            yield f"data: {json.dumps({'status': 'COMPLETE'})}\n\n"
            
        except Exception as e:
            err_msg = f"[CRITICAL ERROR] {e}"
            save_log(video_id, err_msg)
            yield f"data: {json.dumps({'status': 'COMPLETE'})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')
