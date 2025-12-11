from flask import Blueprint, render_template, request, Response, stream_with_context, current_app
import json
import time  # <--- FIXED: Added missing import
import os
import subprocess
import concurrent.futures
from datetime import datetime
from app.services.media_service import MediaService
from app.services.audio_service import AudioService

audit_bp = Blueprint('audit', __name__)

def run_cmd(command):
    """Executes shell command strictly."""
    # Using subprocess.run for blocking calls in threads
    subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def run_script_with_log(command, video_id):
    """Runs a script and streams output (for linear steps)."""
    try:
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in iter(process.stdout.readline, ''):
            clean = line.strip()
            if clean:
                save_log(video_id, clean)
                yield clean
        process.stdout.close()
        process.wait()
    except Exception as e:
        yield f"[ERROR] {e}"

def save_log(video_id, message):
    try:
        log_dir = os.path.join(current_app.config['STORAGE_DIR'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"{video_id}.log"), "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
    except: pass

@audit_bp.route('/report/<video_id>')
def view_report(video_id):
    base_storage = current_app.config['STORAGE_DIR']
    output_dir = os.path.join(current_app.config['BASE_DIR'], 'output_assets', video_id)

    # Helper to safely load JSON
    def load_json(filename, default_type=dict):
        path = os.path.join(output_dir, f"{video_id}_{filename}")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f: return json.load(f)
            except: pass
        return default_type()

    # Load Data
    data = {
        "meta": load_json("details.json", dict),
        "report": load_json("narrative_report.json", dict),
        "transcript": load_json("transcription.json", list),
        "music": load_json("music.json", dict),
        "ai_audio": load_json("ai_audio.json", dict)
    }

    # Media Paths
    media = {
        "vocal": f"/media/storage/audio/{video_id}_vocals.mp3",
        "instr": f"/media/storage/audio/{video_id}_instrumentals.mp3"
    }

    # Frames
    frames_dir = os.path.join(base_storage, "frames", video_id)
    frames = []
    if os.path.exists(frames_dir):
        all_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        frames = [f"/media/storage/frames/{video_id}/{f}" for f in all_frames[::5]]

    return render_template('report.html', video_id=video_id, data=data, media=media, frames=frames)

@audit_bp.route('/audit/stream/<video_id>')
def stream_audit(video_id):
    def generate():
        base_dir = current_app.config['BASE_DIR']
        video_path = f"{base_dir}/storage/videos/{video_id}.mp4"
        audio_path = f"{base_dir}/storage/audio/{video_id}.mp3"

        # Reset Log
        log_dir = os.path.join(current_app.config['STORAGE_DIR'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"{video_id}.log"), 'w') as f: f.write("--- INIT PARALLEL AUDIT ---\n")

        def log(msg, step=None):
            save_log(video_id, msg)
            data = {'message': msg}
            if step is not None: data['step'] = step
            return f"data: {json.dumps(data)}\n\n"

        yield log(f"🚀 Starting Parallel Audit for {video_id}...")

        # --- EXECUTOR FOR PARALLEL TASKS ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:

            # --- TRACK A: METADATA (Immediate) ---
            yield log("Starting Track A: Metadata...", 0)
            # Use separate function calls or lambda to avoid immediate execution
            future_meta = executor.submit(run_cmd, f"python3 python/get_youtube_video_metadata.py --id='{video_id}'")

            # --- START TRACK B: DOWNLOAD & AUDIO EXTRACT ---
            yield log("Starting Track B: Media Download...", 1)
            try:
                # 1. Download Video (Blocking)
                run_cmd(f"python3 python/get_video_file.py --id='{video_id}'")
                yield log("Video downloaded.", 1)

                # 3. Extract Audio (MP3) (Blocking)
                yield log("Extracting Audio...", 2)
                run_cmd(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y")
                yield log("Audio extracted.", 2)

            except Exception as e:
                yield log(f"[CRITICAL] Media Prep Failed: {e}")
                return

            # --- SPLIT POINT: PARALLEL SUB-TRACKS ---
            yield log("🔀 Splitting Pipeline: Processing Audio & Visuals simultaneously...", 3)

            def track_b1_audio():
                try:
                    # 4. LALAL.AI
                    save_log(video_id, "Track B1: Sending to LALAL.ai...")
                    lalal = AudioService.split_stems(audio_path, video_id)
                    if lalal.get("error"):
                        save_log(video_id, f"LALAL Error: {lalal['error']}")
                        # Continue with fallback if needed, but here we raise to notify
                        # raise Exception(lalal['error'])

                    # 7. Transcribe Vocals
                    save_log(video_id, "Track B1: Transcribing Vocals...")
                    run_cmd(f"python3 python/transcribe_video.py --id='{video_id}'")

                    # 9. Music Detection
                    save_log(video_id, "Track B1: Checking Music Copyright...")
                    run_cmd(f"python3 python/detect_music.py {video_id}")

                    # 11. AI Voice & Audibility
                    save_log(video_id, "Track B1: AI Voice Analysis...")
                    run_cmd(f"python3 python/detect_ai_audio.py {video_id}")

                    # Calculate Audibility if LALAL worked
                    if not lalal.get("error"):
                        ratio = AudioService.calculate_audibility(lalal['vocals_path'], lalal['instr_path'])
                        # Save simple JSON
                        out_path = f"output_assets/{video_id}/{video_id}_audibility.json"
                        with open(out_path, "w") as f: json.dump({"percent": ratio}, f)

                    return "Audio Analysis Complete"
                except Exception as e:
                    save_log(video_id, f"Track B1 Failed: {e}")
                    raise e

            def track_b2_visual():
                try:
                    # 5. Extract Frames
                    save_log(video_id, "Track B2: Extracting Frames...")
                    run_cmd(f"python3 python/split_video_into_frames.py --fps 0.33 --files '{video_id}.mp4'")

                    # 6 & 8. AI Vision Analysis & Summary
                    save_log(video_id, "Track B2: AI Vision Analysis...")
                    # Note: You need a script that does JUST the vision part if narrative_analyser does everything
                    # For now, we assume narrative_analyser handles the final convergence,
                    # so we might just do frame extraction here if analyse_frames.py isn't ready.
                    # If you have analyse_frames.py:
                    run_cmd(f"python3 python/analyse_frames.py {video_id}")
                    return "Visual Analysis Complete"
                except Exception as e:
                    save_log(video_id, f"Track B2 Failed: {e}")
                    raise e

            # Launch Threads
            future_audio = executor.submit(track_b1_audio)
            future_visual = executor.submit(track_b2_visual)

            # Loop while running to keep connection alive
            while not (future_audio.done() and future_visual.done()):
                time.sleep(1) # <--- This caused the error before
                yield log("... Processing ...")

            # Check Results
            try:
                future_audio.result()
                yield log("✅ Audio Analysis Completed.", 4)
            except Exception as e:
                yield log(f"❌ Audio Track Failed: {e}")

            try:
                future_visual.result()
                yield log("✅ Visual Analysis Completed.", 5)
            except Exception as e:
                yield log(f"❌ Visual Track Failed: {e}")

        # --- STEP 10: CONVERGENCE ---
        yield log("🏁 Final Convergence: Generating Policy Report...", 6)
        try:
            run_cmd(f"python3 python/narrative_analyser.py {video_id}")
            yield log("✅ Report Generated.", 7)
        except Exception as e:
            yield log(f"❌ Reporting Failed: {e}")

        yield f"data: {json.dumps({'status': 'COMPLETE'})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')
