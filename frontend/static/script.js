document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("fileInput");
  const runBtn = document.getElementById("runBtn"); 
  const statusText = document.getElementById("status");

  runBtn.disabled = true;

  fileInput.addEventListener("change", () => {
    runBtn.disabled = !fileInput.files.length;
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


runBtn.addEventListener("click", () => {
  const config = collectInputs();
  statusText.textContent = "Running analysis with config: " + JSON.stringify(config, null, 2);

});


document.querySelectorAll(".collapsible-header").forEach(header => {
  header.addEventListener("click", () => {
    header.parentElement.classList.toggle("open");
  });
});
});