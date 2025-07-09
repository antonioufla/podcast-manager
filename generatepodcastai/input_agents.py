import os
from typing import List, Optional
from pathlib import Path
from bs4 import BeautifulSoup
import requests
import yt_dlp
from moviepy.editor import AudioFileClip
import tempfile

class TextInputAgent:
    def __init__(self):
        pass

    def load_text_file(self, filepath: str) -> str:
        ext = Path(filepath).suffix.lower()
        if ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.docx':
            from docx import Document
            doc = Document(filepath)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\\n'.join(full_text)
        elif ext == '.pdf':
            import PyPDF2
            text = ''
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + '\\n'
            return text
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

class URLInputAgent:
    def __init__(self):
        pass

    def fetch_url_content(self, url: str) -> str:
        if "youtube.com" in url or "youtu.be" in url:
            # Delegate to VideoInputAgent
            return None
        else:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                texts = soup.stripped_strings
                return ' '.join(texts)
            else:
                raise Exception(f"Failed to fetch URL content: {url}")

class VideoInputAgent:
    def __init__(self):
        pass

    def download_video(self, url: str, output_path: Optional[str] = None) -> str:
        ydl_opts = {
            'format': 'best',
            'outtmpl': output_path or 'downloaded_video.%(ext)s',
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info_dict)
        return filename

    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        output_path = output_path or 'extracted_audio.wav'
        clip = AudioFileClip(video_path)
        clip.audio.write_audiofile(output_path, logger=None)
        clip.close()
        return output_path

    def transcribe_audio(self, audio_path: str) -> str:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result['text']
