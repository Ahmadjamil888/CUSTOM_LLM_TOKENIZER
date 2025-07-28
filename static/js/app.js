document.addEventListener('DOMContentLoaded', function () {
  const uploadForm = document.getElementById('upload-form');
  const fileInput = document.getElementById('file');
  const statusMessage = document.getElementById('status-message');

  uploadForm.addEventListener('submit', async function (e) {
    e.preventDefault();
    
    if (!fileInput.files.length) {
      statusMessage.innerText = "Please select a file.";
      return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    statusMessage.innerText = "Uploading and processing...";

    try {
      const response = await fetch('/upload', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        statusMessage.innerText = "✅ File processed successfully. Check 'processed/output.json'.";
      } else {
        statusMessage.innerText = "❌ Error: " + data.message;
      }
    } catch (error) {
      console.error(error);
      statusMessage.innerText = "❌ Upload failed. Check console for details.";
    }
  });
});
