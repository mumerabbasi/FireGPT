```bash
# Build the base docker file first
docker build -f Dockerfile.base -t ami-base .

# Build the MCP Server (resolves most dependencies)
docker build -f Dockerfile.mcp -t ami-mcp .

# Build the Backend (also contains the frontend)
docker build -f Dockerfile.backend -t ami-backend .

# Build the fire assessment engine
docker build -f Dockerfile.gee -t ami-gee

# (Optional) Build the ollama with the models baked in
docker build -f Dockerfile.ollama -t ami-ollama
```

Once all the builds are done, modify respectively in the docker-compose.yml file and run. If you're using the traditional ollama instead of the build Ollama file in docker-compose (check the first image name specified), make sure you use the interactive shell and pull the required models. Ollama uses qwen2.5vl and qwen3:8b. So to open an interactive shell, perform `docker ps` and find out the container name for ollama, and then run `docker exec -it <containername> sh`. After this perform the following:
```bash
ollama pull qwen2.5vl
ollama pull qwen3:8b
```

For MCP you have to download the models respectively, in our case it's BAAI/bge-m3 and BAAI/bge-reranker-v2-m3. You can actually use the scripts present in utilities to download the models and set the paths via FGPT_EMBED_MODEL environment variable.

Used Environment Variables

| Environment Variable | Default Value                                       | Description                                           |
| :------------------- | :-------------------------------------------------- | :---------------------------------------------------- |
| **FGPT_MCP_URL** | `http://localhost:7790/mcp/`                        | URL for the Model Context Protocol (MCP) server.      |
| **OPENAI_API_BASE** | `http://localhost:11434/v1`                         | Base URL for the OpenAI-compatible API (used by Ollama). |
| **FGPT_MODEL** | `qwen3:8b (primary) / qwen2.5vl (vision)`           | Specifies the language model to be used.              |
| **EE_SERVICE_ACCOUNT_KEY** | `None` (environment variable must be set)     | Path to the Earth Engine service account key.         |
| **FGPT_CHUNK_SIZE** | `1024`                                              | Size of text chunks for document processing.          |
| **FGPT_CHUNK_OVERLAP** | `128`                                             | Overlap between text chunks.                          |
| **FGPT_COLLECTION** | `fire_docs`                                         | Name of the document collection in the vector store.  |
| **FGPT_EMBED_MODEL** | `../../models/bge-m3` or `/app/mcp/models/bge-m3` | Path to the embedding model.                          |
| **FGPT_DB_PATH_SESSION** | `stores/session`                                | Path for the session-specific document store.         |
| **FGPT_DB_PATH_LOCAL** | `stores/local`                                  | Path for the regional/local document store.           |
| **FGPT_DB_PATH_GLOBAL** | `stores/global`                                 | Path for the global document store.                   |
| **FGPT_RERANK_MODEL** | `/app/mcp/models/bge-reranker-v2-m3`             | Path to the reranking model.                          |
| **FGPT_CANDIDATE_K** | `50`                                                | Number of candidate chunks to retrieve.               |
| **FGPT_TOP_K** | `5`                                                 | Number of top chunks to use after reranking.          |
| **FGPT_HOST** | `0.0.0.0`                                           | Host address for the application.                     |
| **FGPT_PORT** | `7790`                                              | Port for the application.                             |
| **FIRESPREAD_API_URL** | `https://api.firefirefire.lol`                    | URL for the fire spread assessment API.               |
| **HF_TOKEN** | `None` (environment variable must be set)         | Hugging Face authentication token.                    |

