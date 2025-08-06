# Ensure you're running this script inside a virtual environment or a shell where Python is available.
#!/bin/bash

python -m pip install --no-cache-dir huggingface_hub

python utilities/download_bge_m3.py
python utilities/download_bge_reranker_v2_m3.py

# Build all docker containers one by one
docker buildx build --push -t kesava89/ami-base:latest -f Dockerfile.base --platform linux/amd64
docker buildx build --push -t kesava89/ami-ollama:latest -f Dockerfile.ollama --platform linux/amd64
docker buildx build --push -t kesava89/ami-mcp:latest -f Dockerfile.mcp --platform linux/amd64
docker buildx build --push -t kesava89/ami-backend:latest -f Dockerfile.backend --platform linux/amd64
docker buildx build --push -t kesava89/ami-gee:latest -f Dockerfile.gee --platform linux/amd64



