from flask import Blueprint, render_template, request, Response, stream_with_context, current_app
import json
import time
import os
import subprocess
import concurrent.futures
from datetime import datetime
from app.services.media_service import MediaService
from app.services.audio_service import AudioService

audit_bp = Blueprint('audit', __name__)

def run_cmd(command):
    """Executes shell command strictly."""
    subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def save_log(video_id, message):
    # Note: Accessing current_app here works if called from main thread,
    # but inside threads we need to be careful. We'll use a hardcoded path relative to run.py for safety in threads.
    try:
        log_dir = os.path.abspath("storage/logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"{video_id}.log"), "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
    except: pass

@audit_bp.route('/report/<video_id>')
def view_report(video_id):
    # (Same as before - omitted for brevity, keep your existing view_report code)
    # ... Use the code from the previous working step ...
    return "Report Page Placeholder (Please restore view_report code)"

@audit_bp.route('/audit/stream/<video_id>')
def stream_audit(video_id):
    def generate():
        # 1. Capture Config in Main Thread (Context Safe)
        base_dir = current_app.config['BASE_DIR']
        lalal_key = current_app.config['LALAL_API_KEY']
        audio_storage_dir = os.path.join(base_dir, "storage/audio")

        video_path = f"{base_dir}/storage/videos/{video_id}.mp4"
        audio_path = f"{base_dir}/storage/audio/{video_id}.mp3"

        # Reset Log
        save_log(video_id, "--- INIT PARALLEL AUDIT ---")

        def log(msg, step=None):
            save_log(video_id, msg)
            data = {'message': msg}
            if step is not None: data['step'] = step
            return f"data: {json.dumps(data)}\n\n"

        yield log(f"🚀 Starting Parallel Audit for {video_id}...")

        # --- EXECUTOR FOR PARALLEL TASKS ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:

            # --- TRACK A: METADATA ---
            yield log("Starting Track A: Metadata...", 0)
            executor.submit(run_cmd, f"python3 python/get_youtube_video_metadata.py --id='{video_id}'")

            # --- TRACK B: MEDIA PREP ---
            yield log("Starting Track B: Media Download...", 1)
            try:
                run_cmd(f"python3 python/get_video_file.py --id='{video_id}'")
                yield log("Video downloaded.", 1)

                yield log("Extracting Audio...", 2)
                run_cmd(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y")
                yield log("Audio extracted.", 2)

            except Exception as e:
                yield log(f"[CRITICAL] Media Prep Failed: {e}")
                return

            # --- PARALLEL SUB-TRACKS ---
            yield log("🔀 Splitting Pipeline: Processing Audio & Visuals simultaneously...", 3)

            def track_b1_audio():
                try:
                    # 4. LALAL.AI (Pass keys explicitly!)
                    save_log(video_id, "Track B1: Sending to LALAL.ai...")
                    lalal = AudioService.split_stems(audio_path, video_id, lalal_key, audio_storage_dir)

                    if lalal.get("error"):
                        save_log(video_id, f"LALAL Error: {lalal['error']}")
                        # Proceed with what we have

                    # 7. Transcribe
                    save_log(video_id, "Track B1: Transcribing...")
                    run_cmd(f"python3 python/transcribe_video.py --id='{video_id}'")

                    # 9. Music Detection
                    save_log(video_id, "Track B1: Checking Music...")
                    run_cmd(f"python3 python/detect_music.py {video_id}")

                    # 11. AI Voice
                    save_log(video_id, "Track B1: AI Voice Analysis...")
                    run_cmd(f"python3 python/detect_ai_audio.py {video_id}")

                    return "Audio Complete"
                except Exception as e:
                    save_log(video_id, f"Track B1 Failed: {e}")
                    raise e

            def track_b2_visual():
                try:
                    # 5. Extract Frames
                    save_log(video_id, "Track B2: Extracting Frames...")
                    run_cmd(f"python3 python/split_video_into_frames.py --fps 0.33 --files '{video_id}.mp4'")

                    # 6. AI Vision (Updated Command)
                    save_log(video_id, "Track B2: AI Vision Analysis...")
                    # Ensure this script creates the necessary reports
                    run_cmd(f"python3 python/analyse_frames.py {video_id}")

                    return "Visual Complete"
                except Exception as e:
                    save_log(video_id, f"Track B2 Failed: {e}")
                    raise e

            # Launch
            future_audio = executor.submit(track_b1_audio)
            future_visual = executor.submit(track_b2_visual)

            while not (future_audio.done() and future_visual.done()):
                time.sleep(1)
                yield log("... Processing ...")

            # Check Results
            if future_audio.exception(): yield log(f"❌ Audio Track Failed: {future_audio.exception()}")
            else: yield log("✅ Audio Analysis Completed.", 4)

            if future_visual.exception(): yield log(f"❌ Visual Track Failed: {future_visual.exception()}")
            else: yield log("✅ Visual Analysis Completed.", 5)

        # --- CONVERGENCE ---
        yield log("🏁 Final Convergence: Generating Policy Report...", 6)
        try:
            run_cmd(f"python3 python/narrative_analyser.py {video_id}")
            yield log("✅ Report Generated.", 7)
        except Exception as e:
            yield log(f"❌ Reporting Failed: {e}")

        yield f"data: {json.dumps({'status': 'COMPLETE'})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')
