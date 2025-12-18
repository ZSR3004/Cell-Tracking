from flask import Flask, request, render_template, jsonify, send_file
import tempfile
import os, sys
import shutil


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.cell_tracking.memory import MemoryManager
import src.cell_tracking.tiffclass as tf
import src.cell_tracking.optical_flow as opt
import src.cell_tracking.heatmap as hm
import src.cell_tracking.kymograph as kg
import src.cell_tracking.saving as s
import src.cell_tracking.vector_magnitude_map as vmh
import src.cell_tracking.raft as r

app = Flask(__name__)

# Create a temporary directory for uploads
temp_dir = tempfile.TemporaryDirectory()
UPLOAD_FOLDER = temp_dir.name
print("Temporary upload folder:", UPLOAD_FOLDER)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_analysis():
    # Get the uploaded file
    uploaded_file = request.files.get('file')
    print("Uploaded file:", uploaded_file)
    if not uploaded_file:
        return "No file uploaded", 400

    # Get JSON config from frontend
    config = request.form.get('config')
    if not config:
        return "No config provided", 400
    import json
    config = json.loads(config)
    print("Config string:", config)


if __name__ == "__main__":
    app.run()