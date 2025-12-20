from flask import Flask, request, render_template, jsonify, send_file, url_for, session
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
import cli.visualization_cli as v
import cli.saving_cli as s
import cli.file_input_cli
import src.cell_tracking.raft as r

app = Flask(__name__)
app.secret_key = "your_super_secret_key_here"

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

    uploaded_file_path = UPLOAD_FOLDER / uploaded_file.filename
    uploaded_file.stream.seek(0)
    uploaded_file.save(uploaded_file_path)

    session['uploaded_file'] = uploaded_file.filename

    s.save_original_video_cli("Original_Video", str(uploaded_file_path), 0)

    video_path = Path("Original_Video.mp4")  
    uploads_dir = Path("static/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest_path = uploads_dir / video_path.name
    shutil.move(video_path, dest_path)

    # return URL for the frontend
    video_url = url_for("static", filename=f"uploads/{dest_path.name}")
    return jsonify({"video_url": video_url})

@app.route('/controls')
def controls():
    return render_template('controls.html')

@app.route('/run_analysis', methods=["POST"])
def run_analysis():
    config = request.get_json()
    filename = session.get('uploaded_file')

    uploaded_file_path = UPLOAD_FOLDER / filename

    my_video = tf.Tiff(str(uploaded_file_path))

    upload_folder_str = str(UPLOAD_FOLDER.resolve())
    # --- Preprocessing ---
    preprocessing_params = {
    "gaussian": {
        "kernel_size": config["preprocessing"]["gaussian"]["kernel_size"],
        "sigma_x": config["preprocessing"]["gaussian"]["sigma_x"]
    },
    "median": {
        "kernel_size": config["preprocessing"]["median"]["kernel_size"]
    },
    "contrast": {
        "alpha": config["preprocessing"]["contrast"]["alpha"],
        "beta": config["preprocessing"]["contrast"]["beta"]
    }
}

    my_video.preprocess_stack(my_video.arr, **preprocessing_params)

    # --- Farneback Optical Flow ---
    farneback_params = {
    "pyr_scale": config["farneback"]["pyr_scale"],
    "levels": config["farneback"]["levels"],
    "win_size": config["farneback"]["win_size"],
    "iterations": config["farneback"]["iterations"],
    "poly_n": config["farneback"]["poly_n"],
    "poly_sigma": config["farneback"]["poly_sigma"]
    }

    # --- RAFT Optical Flow ---
    rfModSize = config["raft"]["model_size"]
    rfModWeight = config["raft"]["model_weights_path"]
    rfGPU = config["raft"]["gpu_flag"]

    # --- Output Options ---
    outKymograph = config["output"]["kymograph"]
    outHeatmap = config["output"]["heatmap"]
    outVectorMag = config["output"]["vector_magnitude"]
    outFarnebackIsolated = config["output"]["farneback_isolated"]
    outFarneback = config["output"]["farneback"]
    outRAFT = config["output"]["raft"]

    if outFarneback:
        combined_flow = opt.calculate_optical_flow(my_video.arr, **farneback_params)
        s.save_flow_cli("farneback", combined_flow, str(UPLOAD_FOLDER))

        if outFarnebackIsolated:
            flow_channel_one = opt.optical_flow(my_video.arr, 1, **farneback_params)
            flow_channel_two = opt.optical_flow(my_video.arr, 2, **farneback_params)

            s.save_original_video_cli("Channel_1", str(uploaded_file_path), 1)

            video_path = Path("Channel_1.mp4")  
            uploads_dir = Path("static/uploads")
            dest_path = uploads_dir / video_path.name
            shutil.move(video_path, dest_path)

            s.save_original_video_cli("Channel_2", str(uploaded_file_path), 1)
            video_path = Path("Channel_2.mp4")  
            uploads_dir = Path("static/uploads")
            dest_path = uploads_dir / video_path.name
            shutil.move(video_path, dest_path)

            
            s.save_flow_cli("farneback_channel_1", flow_channel_one, str(UPLOAD_FOLDER))
            s.save_flow_cli("farneback_channel_2", flow_channel_two, str(UPLOAD_FOLDER))
            

        if outKymograph:
            v.plot_basic_kymo_cli(combined_flow, os.path.join(str(UPLOAD_FOLDER), "kymograph_nuclei_dyed_flow.png"))


        if outHeatmap:
            v.plot_heatmap_cli(combined_flow, "Nuclei Dyed Heatmap",
                               os.path.join(str(UPLOAD_FOLDER), "heatmap_nuclei_dyed_flow.mp4"))
            
        if outVectorMag:
            v.vector_video_cli("vector_field_video", combined_flow)
            

    if outRAFT:
        raft_flow = r.calcOpticalFlowRAFT(my_video, rfModSize, rfModWeight, rfGPU, **preprocessing_params)

        if outKymograph:
            v.plot_basic_kymo_cli(raft_flow, os.path.join(str(UPLOAD_FOLDER), "kymograph_phase_contrast_flow.png"))

        if outHeatmap:
            v.plot_heatmap_cli(raft_flow, "Phase Contrast Heatmap",
                               os.path.join(str(UPLOAD_FOLDER), "heatmap_raft_flow.mp4"))
        if outVectorMag:
            v.vector_video_cli("vector_field_video", raft_flow)
    
    return jsonify({"message": "Analysis finished!"})


if __name__ == "__main__":
    app.run()
