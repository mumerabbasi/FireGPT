/////// Backend URL /////////
const backend_url = "http://127.0.0.1:8000"

// Get map function here /////////////
const mapIframe = document.getElementById('mapIframe');

/////////////////////////////////
// Theme toggle logic
function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
}
// Initialize theme from storage
(function () {
  const saved = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
})();

const fileInput = document.getElementById('fileInput');
const sourcesList = document.getElementById('sourcesList');
const imageInput = document.getElementById('imageInput');
const previewArea = document.getElementById('previewArea');
const chatInput = document.querySelector('.chat-input');
const sendBtn = document.querySelector('.send-btn');
const addDocumentsBtn = document.getElementById('addDocumentsBtn');
const messagesArea = document.getElementById('messagesArea');
let uploadedFiles = [];

// --- Block double send (send button logic) ---
let isWaitingForResponse = false;
// --- Block double upload (Add Documents button logic) ---
let isUploadingDocuments = false;

// Auto-resize textarea up to 5 lines (~130px)
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  const maxHeight = 160; // ~5 lines
  chatInput.style.height = Math.min(chatInput.scrollHeight, maxHeight) + 'px';
  // enable send only if not waiting for response
  if (!isWaitingForResponse) {
    sendBtn.disabled = !chatInput.value.trim() && !imageInput.files.length;
  }
});

imageInput.addEventListener('change', () => {
  previewArea.innerHTML = '';
  Array.from(imageInput.files).forEach(file => {
    const url = URL.createObjectURL(file);
    const img = document.createElement('img');
    img.src = url;
    previewArea.appendChild(img);
  });
  if (!isWaitingForResponse) {
    sendBtn.disabled = false;
  }
});

async function sendMessage() {
  // Prevent duplicate sends
  if (isWaitingForResponse) return;
  isWaitingForResponse = true;
  sendBtn.disabled = true;

  const text = chatInput.value.trim();
  const images = Array.from(imageInput.files);

  // Show user message in UI
  if (text) appendMessage(text, 'user');
  for (const img of images) {
    const reader = new FileReader();
    reader.onload = e => appendImage(e.target.result, 'user');
    reader.readAsDataURL(img);
  }

  // Get Map features
  const map_features = JSON.stringify(mapIframe.contentWindow.extract_map_features_post());
  // extract_map_features_post();

  // Prepare form data
  const formData = new FormData();
  formData.append("message", text);
  images.forEach((file, index) => {
    formData.append("images", file); // FastAPI will receive as a list
  });
  formData.append("map_features", map_features);

  // Reset input UI
  chatInput.value = '';
  imageInput.value = '';
  previewArea.innerHTML = '';
  chatInput.style.height = 'auto';

  // Thinking ... logic
  const assistant_replay = appendMessage(`<span class="typing dots">Thinking</span>`, 'assistant');

  // Fetch 
  try {
    const response = await fetch(backend_url + "/send-chat", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();
    if (response.ok) {
      if (result.reply) {
        changeMessage(assistant_replay, result.reply);
      }
      // Draw the pathList if any 
      if (result.pathList) {
        result.pathList.forEach(coords => {
          mapIframe.contentWindow.drawWaypointPath(coords);
        });
      }
    } else {
      changeMessage(assistant_replay, "Error: " + result.detail);
    }
  } catch (err) {
    console.error("Send failed:", err);
    changeMessage(assistant_replay, "Network error");
  }

  // Reset input UI
  chatInput.value = '';
  imageInput.value = '';
  previewArea.innerHTML = '';
  chatInput.style.height = 'auto';

  // Done waiting, re-enable sendBtn only if there is text or images
  isWaitingForResponse = false;
  sendBtn.disabled = !(chatInput.value.trim() || imageInput.files.length);
}

sendBtn.addEventListener('click', sendMessage);

chatInput.addEventListener('keydown', e => {
  // Allow send only if not waiting for response
  if (e.key === 'Enter' && !e.shiftKey && !isWaitingForResponse && !sendBtn.disabled) {
    e.preventDefault();
    sendMessage();
  }
});

// Append a new message //
function appendMessage(text, role) {
  const msg = document.createElement('div');
  msg.className = `message ${role}`;
  msg.innerHTML = DOMPurify.sanitize(marked.parse(text));
  messagesArea.appendChild(msg);
  messagesArea.scrollTop = messagesArea.scrollHeight;
  return msg;
}

// Just change the message do not append it //
function changeMessage(msg_div, text) {
  msg_div.innerHTML = DOMPurify.sanitize(marked.parse(text));
  messagesArea.scrollTop = messagesArea.scrollHeight;
}

function appendImage(src, role) {
  const msg = document.createElement('div'); msg.className = `message ${role}`;
  const img = document.createElement('img'); img.src = src; msg.appendChild(img);
  messagesArea.appendChild(msg); messagesArea.scrollTop = messagesArea.scrollHeight;
}

async function removeAllDocuments() {
  removeAllDocumentsBtn.disabled = true;
  try {
    const response = await fetch(backend_url + "/delete-all", {
      method: "DELETE",
    });
    if (response.ok) {
      uploadedFiles = [];
      sourcesList.innerHTML = '';
    } else {
      alert('Failed to delete all documents');
    }
  } catch (err) {
    console.error('Error deleting all documents:', err);
    alert('Network error while deleting documents');
  }
  removeAllDocumentsBtn.disabled = false;
}

// Add Documents button disables during upload
fileInput.addEventListener('change', async function () {
  if (isUploadingDocuments) return;
  isUploadingDocuments = true;
  addDocumentsBtn.disabled = true;

  const files = Array.from(this.files);
  for (const file of files) {
    if (!uploadedFiles.includes(file.name)) {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await fetch(backend_url + "/upload", {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          uploadedFiles.push(file.name);
          addSourceItem(file.name);
        } else {
          alert('Upload failed: ' + (await response.text()));
        }
      } catch (err) {
        console.error('Error uploading file:', err);
        alert('Upload failed: Network error');
      }
    }
  }
  fileInput.value = '';
  isUploadingDocuments = false;
  addDocumentsBtn.disabled = false;
});

// Load uploaded files from backend on page load
async function loadUploadedFiles() {
  try {
    const response = await fetch(backend_url + "/list-files");
    if (response.ok) {
      const files = await response.json();
      uploadedFiles = files;
      files.forEach(filename => addSourceItem(filename));
    } else {
      console.error("Failed to fetch file list.");
    }
  } catch (err) {
    console.error("Error fetching uploaded files:", err);
  }
}


function addSourceItem(filename) {
  const div = document.createElement('div');
  div.className = 'doc-item';
  div.textContent = filename;


  const removeBtn = document.createElement('button');
  removeBtn.textContent = '✕';
  removeBtn.className = 'doc-remove-btn';

  removeBtn.onclick = async function () {
    removeBtn.disabled = true;
    try {
      const response = await fetch(`${backend_url}/delete/${encodeURIComponent(filename)}`, {
        method: 'DELETE'
      });
      if (response.ok) {
        uploadedFiles = uploadedFiles.filter(f => f !== filename);
        div.remove();
      } else {
        removeBtn.disabled = false;
        alert(`Failed to delete file: ${filename}`);
      }
    } catch (err) {
      console.error('Error deleting file:', err);
      alert('Network error while deleting file');
    }
  };

  div.appendChild(removeBtn);
  sourcesList.appendChild(div);
}


window.addEventListener("DOMContentLoaded", loadUploadedFiles);