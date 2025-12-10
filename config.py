import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-123')
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Database
    DB_PATH = os.path.join(BASE_DIR, 'video_audit.db')
    
    # Storage Paths
    STORAGE_DIR = os.path.join(BASE_DIR, 'storage')
    VIDEO_DIR = os.path.join(STORAGE_DIR, 'videos')
    AUDIO_DIR = os.path.join(STORAGE_DIR, 'audio')
    FRAMES_DIR = os.path.join(STORAGE_DIR, 'frames')
    STEMS_DIR = os.path.join(STORAGE_DIR, 'stems')
    
    # API Keys
    LALAL_API_KEY = os.getenv('LALAL_API_KEY')
    ACR_ACCESS_KEY = os.getenv('ACR_ACCESS_KEY')
    ACR_ACCESS_SECRET = os.getenv('ACR_ACCESS_SECRET')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Policy File
    POLICY_PATH = os.path.join(BASE_DIR, 'policies.json')

    @staticmethod
    def init_app(app):
        # Ensure directories exist
        for d in [Config.VIDEO_DIR, Config.AUDIO_DIR, Config.FRAMES_DIR, Config.STEMS_DIR]:
            os.makedirs(d, exist_ok=True)
