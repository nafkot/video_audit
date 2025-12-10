from flask import Blueprint, render_template, request, Response, stream_with_context
import json
import time
import os
import glob
from app.services.media_service import MediaService
from app.services.audio_service import AudioService

audit_bp = Blueprint('audit', __name__)

@audit_bp.route('/report/<video_id>')
def view_report(video_id):
    """
    Aggregates all artifacts generated during the audit for display.
    """
    base_storage = current_app.config['STORAGE_DIR']
    output_dir = os.path.join(current_app.config['BASE_DIR'], 'output_assets', video_id)

    # 1. Load Metadata & Report
    metadata = {}
    report = {}
    transcript = []

    # Try loading the final Policy/Narrative Report
    report_path = os.path.join(output_dir, f"{video_id}_narrative_report.json")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)

    # Try loading Metadata
    meta_path = os.path.join(output_dir, f"{video_id}_details.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

    # Try loading Transcript
    trans_path = os.path.join(output_dir, f"{video_id}_transcription.json")
    if os.path.exists(trans_path):
        with open(trans_path, 'r') as f:
            transcript = json.load(f)

    # 2. Find Media Assets
    # Audio Stems (LALAL.ai output)
    vocals_url = f"/media/storage/stems/{video_id}_vocal.mp3"
    instr_url = f"/media/storage/stems/{video_id}_no_vocals.mp3" # or instrumental.mp3

    # Frames (Get list of all .jpg images)
    frames_dir = os.path.join(base_storage, "frames", video_id)
    frames = []
    if os.path.exists(frames_dir):
        # Get all jpgs, sort them, and create URLs
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        # Pick a sample if too many (e.g., every 5th frame) to avoid crashing browser
        frames = [f"/media/storage/frames/{video_id}/{f}" for f in frame_files]

    return render_template(
        'report.html',
        video_id=video_id,
        metadata=metadata,
        report=report,
        transcript=transcript,
        vocals_url=vocals_url,
        instr_url=instr_url,
        frames=frames
    )

@audit_bp.route('/audit/stream/<video_id>')
def stream_audit(video_id):
    def generate():
        yield f"data: {json.dumps({'message': '🚀 Starting Audit...'})}\n\n"
        
        try:
            # 1. Download
            yield f"data: {json.dumps({'step': 1, 'message': 'Downloading Video...'})}\n\n"
            vid_path = MediaService.download_video(video_id)
            
            # 2. Audio
            yield f"data: {json.dumps({'step': 2, 'message': 'Extracting Audio...'})}\n\n"
            aud_path = MediaService.extract_audio(vid_path, video_id)
            
            # 3. Frames
            yield f"data: {json.dumps({'step': 3, 'message': 'Splitting Frames (1/3s)...'})}\n\n"
            frames_dir = MediaService.extract_frames(vid_path, video_id)

            # 4. LALAL.AI
            yield f"data: {json.dumps({'step': 4, 'message': 'Separating Vocals (LALAL.ai)...'})}\n\n"
            stems = AudioService.split_stems(aud_path, video_id)
            
            # 5. Copyright
            yield f"data: {json.dumps({'step': 5, 'message': 'Checking Copyright (ACR)...'})}\n\n"
            music_data = AudioService.detect_copyright(aud_path)
            
            yield f"data: {json.dumps({'status': 'COMPLETE'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

