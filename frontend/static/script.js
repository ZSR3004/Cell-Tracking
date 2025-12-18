const fileInput = document.getElementById("fileInput");

runBtn.disabled = true;

fileInput.addEventListener("change", () => {
  runBtn.disabled = !fileInput.files.length;
});

//const runBtn = document.getElementById("runBtn");
const statusText = document.getElementById("status");

runBtn.addEventListener("click", () => {
  statusText.textContent = "Run clicked (backend not connected)";
});

document.querySelectorAll(".collapsible-header").forEach(header => {
  header.addEventListener("click", () => {
    header.parentElement.classList.toggle("open");
  });
});
