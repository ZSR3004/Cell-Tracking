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
  
  document.addEventListener("DOMContentLoaded", () => {
    const downloadBtn = document.getElementById("downloadBtn");
    downloadBtn.disabled = true;
    const runBtn = document.getElementById("runBtn");
    const statusTextTwo = document.getElementById("status_two");
  
    runBtn.addEventListener("click", async () => {
      runBtn.disabled = true;
      const config = collectInputs();
      statusTextTwo.textContent = "Running analysis...";
  
      const response = await fetch("/run_analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
        credentials: "same-origin"
      });
  
      const data = await response.json();
      statusTextTwo.textContent = data.message;

      const dynamicOutputs = document.getElementById("dynamic-outputs");
      dynamicOutputs.innerHTML = ""; // only clear the videos/images

      for (const [type, url] of Object.entries(data.outputs)) {
        const container = document.createElement("div");
        container.style.marginBottom = "20px";
    
        const label = document.createElement("p");
        label.textContent = type.replace(/_/g, " ").toUpperCase();
        label.style.fontWeight = "bold";
        container.appendChild(label);

        if (url.endsWith(".mp4")) {
          const video = document.createElement("video");
          video.src = url;
          video.controls = true;
          video.width = 500;
          dynamicOutputs.appendChild(video);
        } else {
          const img = document.createElement("img");
          img.src = url;
          img.width = 500;
          dynamicOutputs.appendChild(img);
        }
      }
      downloadBtn.style.display = "inline-block";
      downloadBtn.disabled = false;
    });
  });

  downloadBtn.addEventListener("click", () => {
    downloadBtn.disabled = true;
    window.location.href = "/download_all";
  });