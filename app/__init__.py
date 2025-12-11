import os
from flask import Flask, send_from_directory
from config import Config
import sqlite3

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    Config.init_app(app)

    from app.routes.main import main_bp
    from app.routes.audit import audit_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(audit_bp)

    # --- SERVING FILES (FIXED) ---

    @app.route('/media/audio/<path:filename>')
    def serve_audio(filename):
        # Serves from storage/audio/
        return send_from_directory(app.config['AUDIO_DIR'], filename)

    @app.route('/media/frames/<video_id>/<path:filename>')
    def serve_frames(video_id, filename):
        # Serves from storage/videos/frames/{video_id}/
        frames_dir = os.path.join(app.config['FRAMES_DIR'], video_id)
        return send_from_directory(frames_dir, filename)

    @app.route('/media/output/<path:filename>')
    def serve_output(filename):
        return send_from_directory(os.path.join(app.config['BASE_DIR'], 'output_assets'), filename)

    return app
