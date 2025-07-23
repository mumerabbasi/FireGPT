from huggingface_hub import snapshot_download
import os

# Ensure you have your HF_TOKEN set as an environment variable or pass it directly
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    print("Error: HF_TOKEN environment variable not set. Please set it before running.")
    exit(1)

print("Downloading BAAI/bge-m3...")
snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir="huggingface_models/bge-m3", # Relative path from where you run the script
    local_dir_use_symlinks=False,
    token=hf_token
)
print("BAAI/bge-m3 download complete.")