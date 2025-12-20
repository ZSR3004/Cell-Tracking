document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("fileInput");
  const goBtn = document.getElementById("goBtn");
  const statusText = document.getElementById("status");
  const videoContainer = document.getElementById("UploadedVideoContainer");
  const uploadedVideo = document.getElementById("uploadedVideo");

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

    uploadedVideo.src = data.video_url;
    uploadedVideo.load();
    videoContainer.style.display = "block";
    statusText.textContent = "Upload complete! Video ready.";

    goBtn.style.display = "inline-block";
  });

  goBtn.addEventListener("click", () => {
    window.location.href = "/controls";
  });
});