document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("fileInput");
  const goBtn = document.getElementById("goBtn");
  const statusText = document.getElementById("status");
  const videoContainer = document.getElementById("UploadedVideoContainer");
  const uploadedVideo = document.getElementById("uploadedVideo");

  // Initially hide video and go button
  videoContainer.style.display = "none";
  goBtn.style.display = "none";

  fileInput.addEventListener("change", async () => {
    statusText.textContent = "Uploading video...";
    
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const response = await fetch("/upload", {
      method: "POST",
      body: formData
    });

    if (!response.ok) throw new Error("Upload failed");

    const data = await response.json();

    // Show uploaded video
    uploadedVideo.src = data.video_url;
    uploadedVideo.load();
    videoContainer.style.display = "block";
    statusText.textContent = "Upload complete! Video ready.";

      // Show Go button
    goBtn.style.display = "inline-block";
  });

  goBtn.addEventListener("click", () => {
    // Show the rest of the controls, or navigate to another page
    window.location.href = "/process";
  });
});

function collectInputs() {
  //--- Preprocessing ---
  const preprocessing = {
    gaussian: {
      kernel_size: Number(document.getElementById("gaussK").value),
      sigma_x: Number(document.getElementById("gaussSigma").value)
    },
    median: {
      kernel_size: Number(document.getElementById("medianK").value)
    },
    contrast: {
      alpha: Number(document.getElementById("contrastAlpha").value),
      beta: Number(document.getElementById("contrastBeta").value)
    }
  };

  // --- Farneback Optical Flow ---
  const farneback = {
    pyr_scale: Number(document.getElementById("fbPyrScale").value),
    levels: Number(document.getElementById("fbLevels").value),
    win_size: Number(document.getElementById("fbWinSize").value),
    iterations: Number(document.getElementById("fbIterations").value),
    poly_n: Number(document.getElementById("fbPolyN").value),
    poly_sigma: Number(document.getElementById("fbPolySigma").value)
  };

  // --- RAFT Optical Flow ---
  const raft = {
    model_size: Number(document.getElementById("rfModSize").value),
    model_weights_path: document.getElementById("rfModWeight").value,
    gpu_flag: document.getElementById("rfGPU").checked
  };

  // --- Output options ---
  const output = {
    kymograph: document.getElementById("outKymograph").checked,
    heatmap: document.getElementById("outHeatmap").checked,
    vector_magnitude: document.getElementById("outVectorMag").checked,
    farneback_isolated: document.getElementById("outFarnebackIsolated").checked,
    farneback: document.getElementById("outFarneback").checked,
    raft: document.getElementById("outRAFT").checked
  };

  // --- Combine everything ---
  const config = {
    preprocessing,
    farneback,
    raft,
    output
  };

  return config;
}

document.querySelectorAll(".collapsible-header").forEach(header => {
  header.addEventListener("click", () => {
    header.closest(".collapsible").classList.toggle("open");
  });
});

runBtn.addEventListener("Processing Parameters", async () => {
  const statusTextTwo = document.getElementById("status_two");


});