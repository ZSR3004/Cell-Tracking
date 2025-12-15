const runBtn = document.getElementById("runBtn");
const statusText = document.getElementById("status");

runBtn.addEventListener("click", () => {
  statusText.textContent = "Run clicked (backend not connected)";
});
