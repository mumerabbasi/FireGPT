# Use an official Python runtime as a parent image
FROM python:3.11-bookworm

# Set the working directory in the container
WORKDIR /app/mcp

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install any needed Python packages specified in requirements.txt
# Using --no-cache-dir to save space
RUN pip install --no-cache-dir -r requirements.txt

# --- Hugging Face Model Downloads ---
# Create the models directory
RUN mkdir -p models
# Set the Hugging Face token as an environment variable
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Download BAAI/bge-m3
RUN export HF_TOKEN && \
    huggingface-cli login --token $HF_TOKEN
RUN huggingface-cli download "BAAI/bge-m3" \
    --local-dir models/bge-m3 \
    --local-dir-use-symlinks False

# Download BAAI/bge-reranker-v2-m3
RUN huggingface-cli download "BAAI/bge-reranker-v2-m3" \
    --local-dir models/bge-reranker-v2-m3 \
    --local-dir-use-symlinks False
# --- End Hugging Face Model Downloads ---

COPY . /app/mcp

# ENVIRONMENT VARIABLES
ENV FGPT_DB_PATH_SESSION=stores/session
ENV FGPT_DB_PATH_LOCAL=stores/local
ENV FGPT_DB_PATH_GLOBAL=stores/global

ENV FGPT_EMBED_MODEL=models/bge-base-en-v1.5
ENV FGPT_RERANK_MODEL=models/bge-reranker-base
ENV FGPT_COLLECTION=fire_docs
ENV CANDIDATE_K=50
ENV FGPT_TOP_K=5

ENV FGPT_HOST=0.0.0.0
ENV FGPT_PORT=7790

EXPOSE $FGPT_PORT

# Command to run the application using Uvicorn
# The --host 0.0.0.0 makes the server accessible from outside the container
CMD ["python", "mcp_server.py"]