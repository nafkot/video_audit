from flask import Blueprint, render_template, request, Response, stream_with_context, current_app, send_file
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
        # Use absolute path to avoid thread context issues
        base_dir = os.path.abspath(os.path.dirname(__file__))
        # Go up two levels to reach root (app/routes -> app -> root)
        root_dir = os.path.dirname(os.path.dirname(base_dir))
        log_dir = os.path.join(root_dir, 'storage', 'logs')

        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"{video_id}.log"), "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
    except Exception as e:
        print(f"Log Error: {e}")

# --- FIXED: Full Report View Function ---
@audit_bp.route('/report/<video_id>')
def view_report(video_id):
    # Use current_app only in main thread
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

@audit_bp.route('/audit/logs/<video_id>')
def download_logs(video_id):
    log_dir = os.path.join(current_app.config['STORAGE_DIR'], 'logs')
    return send_file(os.path.join(log_dir, f"{video_id}.log"), mimetype='text/plain')

@audit_bp.route('/audit/stream/<video_id>')
def stream_audit(video_id):
    def generate():
        # Capture config in main thread
        base_dir = current_app.config['BASE_DIR']
        lalal_key = current_app.config['LALAL_API_KEY']
        audio_storage_dir = os.path.join(base_dir, "storage/audio")

        video_path = f"{base_dir}/storage/videos/{video_id}.mp4"
        audio_path = f"{base_dir}/storage/audio/{video_id}.mp3"

        save_log(video_id, "--- INIT PARALLEL AUDIT ---")

        # FIX: Renamed 'step' to 'step_index' to match index.html JavaScript
        def log(msg, step_idx=None):
            save_log(video_id, msg)
            data = {'message': msg}
            if step_idx is not None: data['step_index'] = step_idx
            return f"data: {json.dumps(data)}\n\n"

        yield log(f"🚀 Starting Parallel Audit for {video_id}...")

        # --- EXECUTOR FOR PARALLEL TASKS ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:

            # --- TRACK A: METADATA ---
            # Step Index 0: "Metadata"
            yield log("Starting Track A: Metadata...", 0)
            executor.submit(run_cmd, f"python3 python/get_youtube_video_metadata.py --id='{video_id}'")

            # --- START TRACK B: DOWNLOAD & AUDIO EXTRACT ---
            # Step Index 1: "Download"
            yield log("Starting Track B: Media Download...", 1)
            try:
                run_cmd(f"python3 python/get_video_file.py --id='{video_id}'")
                yield log("Video downloaded.", 1)

                # Step Index 3: "Audio Extract" (Skipping 2 "Frame Split" for now)
                yield log("Extracting Audio...", 3)
                run_cmd(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y")
                yield log("Audio extracted.", 3)

            except Exception as e:
                yield log(f"[CRITICAL] Media Prep Failed: {e}")
                return

            # --- SPLIT POINT: PARALLEL SUB-TRACKS ---
            yield log("🔀 Splitting Pipeline: Audio & Visuals...", 3)

            def track_b1_audio():
                try:
                    # Step Index 4: "LALAL.ai Split"
                    save_log(video_id, "Track B1: Sending to LALAL.ai...")
                    lalal = AudioService.split_stems(audio_path, video_id, lalal_key, audio_storage_dir)

                    if lalal.get("error"):
                        save_log(video_id, f"LALAL Error: {lalal['error']}")

                    # Transcribe
                    save_log(video_id, "Track B1: Transcribing...")
                    run_cmd(f"python3 python/transcribe_video.py --id='{video_id}'")

                    # Step Index 5: "Music ID"
                    save_log(video_id, "Track B1: Checking Music...")
                    run_cmd(f"python3 python/detect_music.py {video_id}")

                    # AI Voice
                    save_log(video_id, "Track B1: AI Voice Analysis...")
                    run_cmd(f"python3 python/detect_ai_audio.py {video_id}")

                    # Audibility
                    if not lalal.get("error"):
                        ratio = AudioService.calculate_audibility(lalal['vocals_path'], lalal['instr_path'])
                        with open(f"output_assets/{video_id}/{video_id}_audibility.json", "w") as f:
                             json.dump({"percent": ratio}, f)

                    return "Audio Complete"
                except Exception as e:
                    save_log(video_id, f"Track B1 Failed: {e}")
                    raise e

            def track_b2_visual():
                try:
                    # Step Index 2: "Frame Split" (We run this in parallel now)
                    save_log(video_id, "Track B2: Extracting Frames...")
                    # Note: We send this log *without* UI update first to not confuse order,
                    # or we can update step 2 now if we want.
                    run_cmd(f"python3 python/split_video_into_frames.py --fps 0.33 --files '{video_id}.mp4'")

                    # Step Index 6: "Visual Analysis"
                    save_log(video_id, "Track B2: AI Vision Analysis...")
                    run_cmd(f"python3 python/analyse_frames.py {video_id}")

                    return "Visual Complete"
                except Exception as e:
                    save_log(video_id, f"Track B2 Failed: {e}")
                    raise e

            # Launch Threads
            future_audio = executor.submit(track_b1_audio)
            future_visual = executor.submit(track_b2_visual)

            # Wait loop
            while not (future_audio.done() and future_visual.done()):
                time.sleep(1)
                yield log("... Processing ...")

            # Check Results and mark steps done
            if future_audio.exception(): yield log(f"❌ Audio Failed: {future_audio.exception()}")
            else: yield log("✅ Audio Analysis Done.", 5) # Mark Music ID/Audio steps done

            if future_visual.exception(): yield log(f"❌ Visual Failed: {future_visual.exception()}")
            else: yield log("✅ Visual Analysis Done.", 6) # Mark Visual step done

        # --- STEP 10: CONVERGENCE ---
        # Step Index 7: "Report Gen"
        yield log("🏁 Final Convergence: Generating Report...", 7)
        try:
            run_cmd(f"python3 python/narrative_analyser.py {video_id}")
            yield log("✅ Report Generated.")
        except Exception as e:
            yield log(f"❌ Reporting Failed: {e}")

        yield f"data: {json.dumps({'status': 'COMPLETE'})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')
