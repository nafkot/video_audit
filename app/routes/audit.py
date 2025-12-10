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
    try:
        log_dir = os.path.join(current_app.config['STORAGE_DIR'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"{video_id}.log"), "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
    except: pass

def run_script(command, video_id):
    try:
        save_log(video_id, f"EXEC: {command}")
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in iter(process.stdout.readline, ''):
            clean = line.strip()
            if clean:
                save_log(video_id, clean)
                yield clean
        process.wait()
        if process.returncode != 0:
            yield f"[ERROR] Script failed with code {process.returncode}"
    except Exception as e:
        yield f"[ERROR] Execution failed: {e}"

@audit_bp.route('/report/<video_id>')
def view_report(video_id):
    base_storage = current_app.config['STORAGE_DIR']
    output_dir = os.path.join(current_app.config['BASE_DIR'], 'output_assets', video_id)

    # --- HELPER: Safely Load JSON with Correct Defaults ---
    def load_json(filename, default_type=dict):
        path = os.path.join(output_dir, f"{video_id}_{filename}")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f: return json.load(f)
            except: pass
        return default_type() # Returns [] for list, {} for dict

    # 1. Load Data (Explicitly set transcript default to LIST)
    data = {
        "meta": load_json("details.json", dict),
        "report": load_json("narrative_report.json", dict),
        "transcript": load_json("transcription.json", list), # <--- FIXED
        "music": load_json("music.json", dict),
        "ai_audio": load_json("ai_audio.json", dict)
    }

    # 2. Media Paths
    media = {
        "vocal": f"/media/storage/stems/{video_id}_vocal.mp3",
        "instr": f"/media/storage/stems/{video_id}_no_vocals.mp3"
    }

    # 3. Frames
    frames_dir = os.path.join(base_storage, "frames", video_id)
    frames = []
    if os.path.exists(frames_dir):
        all_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        frames = [f"/media/storage/frames/{video_id}/{f}" for f in all_frames[::5]]

    return render_template(
        'report.html',
        video_id=video_id,
        data=data,
        media=media,
        frames=frames
    )

@audit_bp.route('/audit/logs/<video_id>')
def download_logs(video_id):
    log_dir = os.path.join(current_app.config['STORAGE_DIR'], 'logs')
    return send_file(os.path.join(log_dir, f"{video_id}.log"), mimetype='text/plain')

@audit_bp.route('/audit/stream/<video_id>')
def stream_audit(video_id):
    def generate():
        # Clear log
        log_dir = os.path.join(current_app.config['STORAGE_DIR'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"{video_id}.log"), 'w') as f: f.write("--- INIT ---\n")

        yield f"data: {json.dumps({'message': '🚀 Starting Audit...'})}\n\n"

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
            yield f"data: {json.dumps({'message': 'Audio extracted.'})}\n\n"

            # 5. Transcription (ADDED THIS STEP)
            yield f"data: {json.dumps({'step': 4, 'message': 'Transcribing Audio...'})}\n\n"
            # Ensure transcribe_video.py exists (I provided it in the previous turn)
            for log in run_script(f"python3 python/transcribe_video.py --id='{video_id}'", video_id):
                yield f"data: {json.dumps({'message': log})}\n\n"

            # 6. LALAL.AI
            yield f"data: {json.dumps({'step': 5, 'message': 'Separating Vocals (LALAL)...'})}\n\n"
            stems = AudioService.split_stems(aud_path, video_id)
            if stems.get("error"):
                 yield f"data: {json.dumps({'message': '[Error] ' + stems['error']})}\n\n"
            else:
                 yield f"data: {json.dumps({'message': 'Stems separated.'})}\n\n"

            # 7. Music ID
            yield f"data: {json.dumps({'step': 6, 'message': 'Identifying Music...'})}\n\n"
            for log in run_script(f"python3 python/detect_music.py {video_id}", video_id):
                yield f"data: {json.dumps({'message': log})}\n\n"

            # 8. AI Voice
            yield f"data: {json.dumps({'step': 6, 'message': 'Checking AI Voice...'})}\n\n"
            for log in run_script(f"python3 python/detect_ai_audio.py {video_id}", video_id):
                yield f"data: {json.dumps({'message': log})}\n\n"

            # 9. Final Report
            yield f"data: {json.dumps({'step': 7, 'message': 'Generating Final Report...'})}\n\n"
            for log in run_script(f"python3 python/narrative_analyser.py {video_id}", video_id):
                yield f"data: {json.dumps({'message': log})}\n\n"

            yield f"data: {json.dumps({'status': 'COMPLETE'})}\n\n"

        except Exception as e:
            save_log(video_id, f"CRITICAL: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')
