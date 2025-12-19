from flask import Flask, request, render_template, jsonify, send_file, url_for
import tempfile
import os, sys
import shutil
from pathlib import Path


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.cell_tracking.memory import MemoryManager
import src.cell_tracking.tiffclass as tf
import src.cell_tracking.optical_flow as opt
import src.cell_tracking.heatmap as hm
import src.cell_tracking.kymograph as kg
import cli.saving_cli as s
import cli.file_input_cli
import src.cell_tracking.vector_magnitude_map as vmh
import src.cell_tracking.raft as r

app = Flask(__name__)

# Create a temporary directory for uploads
temp_dir = tempfile.TemporaryDirectory()
UPLOAD_FOLDER = Path(__file__).parent / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def file_upload():
    # Get the uploaded file
    uploaded_file = request.files.get('file')

    if not uploaded_file:
        return "No file uploaded", 400
    
    tiff_path = UPLOAD_FOLDER / uploaded_file.filename
    uploaded_file.save(tiff_path)

    uploaded_file_path = UPLOAD_FOLDER / uploaded_file.filename
    uploaded_file.stream.seek(0)
    uploaded_file.save(uploaded_file_path)

    s.save_original_video_cli("Original_Video", str(uploaded_file_path), 0)

    video_path = Path("Original_Video.mp4")  
    uploads_dir = Path("static/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest_path = uploads_dir / video_path.name
    shutil.move(video_path, dest_path)

    # return URL for the frontend
    video_url = url_for("static", filename=f"uploads/{dest_path.name}")
    return jsonify({"video_url": video_url})

@app.route('/process')
def controls():
    return render_template('controls.html')

def get_controls():
    return;


if __name__ == "__main__":
    app.run()
