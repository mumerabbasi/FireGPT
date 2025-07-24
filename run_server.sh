# Create very small venv 
python3 -m venv --clear --system-site-packages --without-pip --prompt "tiny" venv

# Install pip and huggingface_hub
source venv/bin/activate
python -m ensurepip
pip install --no-cache-dir huggingface_hub

# Run the python script from utilities folder
export HF_TOKEN="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"
python utilities/download_bge_m3.py
python utilities/download_bge_reranker_v2_m3.py

# Run the docker compose command
docker compose -f docker-compose.yml up

