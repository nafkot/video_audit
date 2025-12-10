import os
import subprocess
import json
from flask import current_app

class MediaService:
    @staticmethod
    def download_video(video_id):
        """Downloads video using yt-dlp."""
        output_path = os.path.join(current_app.config['VIDEO_DIR'], f"{video_id}.mp4")
        
        if os.path.exists(output_path):
            return output_path

        cmd = [
            "yt-dlp", "-f", "best[ext=mp4]", 
            "-o", output_path, 
            f"https://www.youtube.com/watch?v={video_id}"
        ]
        subprocess.run(cmd, check=True)
        return output_path

    @staticmethod
    def extract_audio(video_path, video_id):
        """Extracts MP3 from MP4."""
        output_path = os.path.join(current_app.config['AUDIO_DIR'], f"{video_id}.mp3")
        
        cmd = [
            "ffmpeg", "-i", video_path, 
            "-q:a", "0", "-map", "a", 
            "-y", output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path

    @staticmethod
    def extract_frames(video_path, video_id, interval=3):
        """Extracts frames every X seconds."""
        output_dir = os.path.join(current_app.config['FRAMES_DIR'], video_id)
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-i", video_path, 
            "-vf", f"fps=1/{interval}", 
            os.path.join(output_dir, "frame_%04d.jpg")
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_dir
