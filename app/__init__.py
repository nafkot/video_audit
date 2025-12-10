import os
from flask import Flask, send_from_directory
from config import Config
import sqlite3

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    Config.init_app(app)

    # Register Blueprints
    from app.routes.main import main_bp
    from app.routes.audit import audit_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(audit_bp)

    # --- NEW: Routes to serve generated content ---
    @app.route('/media/storage/<path:filename>')
    def storage_files(filename):
        """Serves raw media (frames, audio) from the storage folder."""
        return send_from_directory(app.config['STORAGE_DIR'], filename)

    @app.route('/media/output/<path:filename>')
    def output_files(filename):
        """Serves reports and assets from the output directory."""
        return send_from_directory(os.path.join(app.config['BASE_DIR'], 'output_assets'), filename)

    return app
