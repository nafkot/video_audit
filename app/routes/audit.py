from flask import Blueprint, render_template, request, Response, stream_with_context, current_app, send_file
import json
import time
import os
import subprocess
import concurrent.futures
import queue
import threading
from datetime import datetime
from app.services.audio_service import AudioService

audit_bp = Blueprint('audit', __name__)

# --- QUEUE-BASED LOGGER ---
class StreamLogger:
    def __init__(self, msg_queue, video_id, storage_dir, debug_mode=False):
        self.queue = msg_queue
        self.video_id = video_id
        self.debug = debug_mode
        self.log_file = os.path.join(storage_dir, 'logs', f"{video_id}.log")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log(self, message, step_idx=None):
        """Logs to file and puts message in queue for UI."""
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        full_msg = f"{timestamp} {message}"
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(full_msg + "\n")
        except: pass
            
        data = {'message': message}
        if step_idx is not None: data['step_index'] = step_idx
        self.queue.put(f"data: {json.dumps(data)}\n\n")

    def run_cmd(self, command):
        """Runs command, streaming output to queue."""
        self.log(f"EXEC: {command}")
        
        process = subprocess.Popen(
            command, shell=True, 
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, bufsize=1, universal_newlines=True
        )
        
        for line in iter(process.stdout.readline, ''):
            clean = line.strip()
            if clean:
                try:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(f"    {clean}\n")
                except: pass

                if self.debug or "error" in clean.lower():
                    self.queue.put(f"data: {json.dumps({'message': '  > ' + clean})}\n\n")

        process.stdout.close()
        process.wait()
        
        if process.returncode != 0:
            raise Exception(f"Command failed (Exit {process.returncode})")

# --- RESTORED REPORT VIEW ---
@audit_bp.route('/report/<video_id>')
def view_report(video_id):
    base_storage = current_app.config['STORAGE_DIR']
    output_dir = os.path.join(current_app.config['BASE_DIR'], 'output_assets', video_id)
    
    def load_json(filename, default_type=dict):
        path = os.path.join(output_dir, f"{video_id}_{filename}")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f: return json.load(f)
            except: pass
        return default_type()
    
    data = {
        "meta": load_json("details.json", dict),
        "report": load_json("narrative_report.json", dict),
        "transcript": load_json("transcription.json", list),
        "music": load_json("music.json", dict),
        "ai_audio": load_json("ai_audio.json", dict),
        "audibility": load_json("audibility.json", dict)
    }

    media = {
        "vocal": f"/media/audio/{video_id}_vocals.mp3",
        "instr": f"/media/audio/{video_id}_instrumentals.mp3"
    }

    frames_dir = os.path.join(base_storage, "frames", video_id)
    frames = []
    if os.path.exists(frames_dir):
        all_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        frames = [f"/media/frames/{video_id}/{f}" for f in all_frames[::5]]

    return render_template('report.html', video_id=video_id, data=data, media=media, frames=frames)

@audit_bp.route('/audit/stream/<video_id>')
def stream_audit(video_id):
    debug_mode = request.args.get('debug') == 'true'
    
    # 1. Capture Config in Main Thread (Context Safe)
    base_dir = current_app.config['BASE_DIR']
    storage_dir = current_app.config['STORAGE_DIR']
    output_dir_base = os.path.join(base_dir, 'output_assets')
    lalal_key = current_app.config['LALAL_API_KEY']
    audio_storage_dir = os.path.join(storage_dir, "audio")
    
    video_path = os.path.join(storage_dir, "videos", f"{video_id}.mp4")
    audio_path = os.path.join(storage_dir, "audio", f"{video_id}.mp3")

    msg_queue = queue.Queue()
    
    # Clear old log
    try: os.remove(os.path.join(storage_dir, 'logs', f"{video_id}.log"))
    except: pass
    
    logger = StreamLogger(msg_queue, video_id, storage_dir, debug_mode)

    def background_task():
        try:
            logger.log(f"🚀 Starting Audit (Debug: {debug_mode})")

            # --- TRACK A: METADATA ---
            logger.log("Step 1: Fetching Metadata...", 0)
            logger.run_cmd(f"python3 python/get_youtube_video_metadata.py --id='{video_id}'")

            # --- TRACK B: PREP ---
            logger.log("Step 2: Downloading Video...", 1)
            logger.run_cmd(f"python3 python/get_video_file.py --id='{video_id}'")
            
            logger.log("Step 3: Extracting Audio...", 2) # Updated Index
            # Ensure dir exists
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            logger.run_cmd(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y")

            # --- PARALLEL TRACKS ---
            logger.log("🔀 Starting Parallel Processing...", 2)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                
                def run_audio_track():
                    try:
                        # 1. LALAL.ai
                        logger.log("Track B1: LALAL.ai Splitting...", 3) # Updated Index
                        def lalal_callback(msg):
                            logger.log(f"  [LALAL] {msg}")

                        lalal = AudioService.split_stems(
                            audio_path, video_id, lalal_key, audio_storage_dir, 
                            log_callback=lalal_callback
                        )
                        
                        vocals_file = audio_path
                        if lalal.get("error"):
                            logger.log(f"LALAL Failed: {lalal['error']}")
                        else:
                            vocals_file = lalal['vocals_path']
                            ratio = AudioService.calculate_audibility(lalal['vocals_path'], lalal['instr_path'])
                            
                            # Save Audibility JSON
                            out_p = os.path.join(output_dir_base, video_id, f"{video_id}_audibility.json")
                            os.makedirs(os.path.dirname(out_p), exist_ok=True)
                            with open(out_p, "w") as f:
                                json.dump({"percent": ratio}, f)

                        # 2. Transcribe
                        logger.log("Track B1: Transcribing...")
                        logger.run_cmd(f"python3 python/transcribe_video.py --id='{video_id}' --file='{vocals_file}'")

                        # 3. Music & AI
                        logger.log("Track B1: Checking Music & AI Voice...", 5) # Updated Index
                        logger.run_cmd(f"python3 python/detect_music.py {video_id}")
                        logger.run_cmd(f"python3 python/detect_ai_audio.py {video_id}")
                        
                        return "Audio Done"
                    except Exception as e:
                        logger.log(f"❌ Audio Track Error: {e}")
                        raise e

                def run_visual_track():
                    try:
                        logger.log("Track B2: Splitting Frames...", 4) # Updated Index
                        logger.run_cmd(f"python3 python/split_video_into_frames.py --fps 0.33 --files '{video_id}.mp4'")
                        
                        logger.log("Track B2: Analyzing Visuals...", 6) # Updated Index
                        logger.run_cmd(f"python3 python/analyse_frames.py {video_id}")
                        return "Visual Done"
                    except Exception as e:
                        logger.log(f"❌ Visual Track Error: {e}")
                        raise e

                # Submit
                f1 = executor.submit(run_audio_track)
                f2 = executor.submit(run_visual_track)
                
                # Wait
                f1.result()
                f2.result()
                logger.log("✅ Analysis Tracks Complete.")

            # --- CONVERGENCE ---
            logger.log("🏁 Generating Final Report...", 7) # Updated Index
            logger.run_cmd(f"python3 python/narrative_analyser.py {video_id}")
            
            logger.log("✅ AUDIT COMPLETE.")
            msg_queue.put(f"data: {json.dumps({'status': 'COMPLETE'})}\n\n")

        except Exception as e:
            logger.log(f"CRITICAL FAILURE: {e}")
            msg_queue.put(f"data: {json.dumps({'error': str(e)})}\n\n")
        
        msg_queue.put(None)

    threading.Thread(target=background_task).start()

    def event_stream():
        while True:
            msg = msg_queue.get()
            if msg is None: break
            yield msg

    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')
