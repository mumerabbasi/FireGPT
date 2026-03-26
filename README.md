<div align="center">

# FireGPT

### AI-Powered Wildfire Intelligence & Drone Operations Platform

An agentic RAG system that combines LLM-based reasoning with real-time geospatial analysis, fire danger modeling, and autonomous drone waypoint generation for wildfire incident response.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct_Agent-1C3C3C?logo=langchain&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Qwen3_8B-000000?logo=ollama&logoColor=white)
![Google Earth Engine](https://img.shields.io/badge/Google_Earth_Engine-Geospatial-4285F4?logo=google-earth&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-Tool_Integration-8B5CF6)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00)

</div>

---

## Demo

<video src="https://github.com/mumerabbasi/FireGPT/raw/main/assets/FireGPT_Demo.mp4" controls width="100%"></video>

---

## What It Does

FireGPT is an end-to-end wildfire decision-support system. A user marks a fire-affected area on an interactive map, uploads relevant operational documents, and converses with an AI agent that can:

1. **Assess fire danger** across the marked region using satellite imagery, weather data, vegetation density, and land cover classification from Google Earth Engine
2. **Retrieve grounded evidence** from uploaded SOPs, safety manuals, and best-practice documents using a multi-tiered RAG pipeline with cross-encoder reranking
3. **Generate drone waypoints** for surveillance, water spraying, search-and-rescue, and post-fire assessment based on the fire danger scores and terrain analysis
4. **Analyze uploaded images** using a vision LLM to detect fire perimeters, attack routes, water sources, and helipads

All reasoning happens through an autonomous agent loop that decides which tools to invoke, in what order, and how to combine their outputs into an actionable response.

---

## Key Features

| Feature | Description |
|:--------|:------------|
| **Agentic Reasoning** | LangGraph ReAct agent with up to 15 autonomous tool-use steps per turn that plans, retrieves, analyzes, and synthesizes without manual orchestration |
| **Real-Time Fire Danger Scoring** | Per-cell danger scores (0 to 100) computed from 5+ GEE datasets: Hansen tree cover, CORINE land cover, GFS weather forecasts, SRTM elevation, and JRC forest maps |
| **Multi-Tiered RAG** | Three-level knowledge base (session / regional / global) with BGE-M3 embeddings and BGE-Reranker-V2-M3 cross-encoder reranking for high-precision retrieval (see below) |
| **MCP Tool Integration** | Retrieval tools and geospatial APIs exposed via Model Context Protocol. The agent discovers and invokes them at runtime through a standardized tool interface |
| **Drone Waypoint Generation** | Generates optimized flight paths (8 to 16 waypoints) for surveillance, fire suppression, and search-and-rescue based on danger scores and terrain |
| **Interactive Map UI** | Leaflet.js map with draw tools for marking fire areas and POIs, geocoding search, and real-time waypoint visualization |
| **Multimodal Input** | Accepts text queries, PDF document uploads, and image uploads analyzed by a vision LLM (Qwen 2.5 VL) |
| **Critical Infrastructure Detection** | Identifies hospitals, schools, fire stations, and shelters near fire zones via OpenStreetMap Overpass API with distance-to-risk calculations |
| **Fully Containerized** | Five Docker services orchestrated via Compose with GPU acceleration support. One command to deploy |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Browser UI                               │
│   ┌─────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│   │  Document    │  │   Chat Interface │  │   Leaflet.js Map  │  │
│   │  Sidebar     │  │   (Streaming)    │  │   + Draw Tools    │  │
│   └─────────────┘  └──────────────────┘  └───────────────────┘  │
└──────────────────────────────┬───────────────────────────────────┘
                               │ REST / SSE
                               ▼
              ┌────────────────────────────────┐
              │     Backend  (FastAPI :8000)    │
              │  • LangGraph ReAct Agent       │
              │  • Session & Document Mgmt     │
              │  • Vision LLM Image Captioning │
              └──────┬────────────────┬────────┘
                     │                │
          MCP (HTTP) │                │ OpenAI-compat API
                     ▼                ▼
     ┌───────────────────┐   ┌──────────────────┐
     │  MCP Server :7790 │   │  Ollama :11434   │
     │  ┌─────────────┐  │   │  • Qwen 3 8B     │
     │  │ RAG Engine  │  │   │  • Qwen 2.5 VL   │
     │  │ (ChromaDB)  │  │   └──────────────────┘
     │  ├─────────────┤  │
     │  │ Session     │  │
     │  │ Local       │  │
     │  │ Global      │  │
     │  └─────────────┘  │
     │  ┌─────────────┐  │
     │  │ FireGEE     │  │
     │  │ Client      │──┼───► Google Earth Engine APIs
     │  └─────────────┘  │    (GFS, Hansen, CORINE, SRTM, JRC)
     └───────────────────┘
```

**Data flow:** User draws a fire area on the map and sends a query → Backend feeds the query + map geometry to the ReAct agent → Agent autonomously invokes MCP tools (fire danger assessment, document retrieval) → MCP server queries GEE for geospatial data and ChromaDB for relevant documents → Agent synthesizes findings into an actionable response streamed back to the UI.

---

## Tech Stack

| Layer | Technologies |
|:------|:-------------|
| **Agent Framework** | LangGraph (ReAct pattern), LangChain, Model Context Protocol (MCP) |
| **Language Models** | Qwen 3 8B (reasoning), Qwen 2.5 VL (vision), served locally via Ollama |
| **Vector Search** | ChromaDB + BGE-M3 embeddings (384-dim) + BGE-Reranker-V2-M3 cross-encoder |
| **Geospatial** | Google Earth Engine, OpenStreetMap Overpass API, GeoPy |
| **Backend** | Python 3.11, FastAPI, SSE-Starlette, HTTPX, uvicorn |
| **Frontend** | HTML/CSS/JS, Leaflet.js 1.9.4 + Leaflet-Draw, Marked.js, DOMPurify |
| **ML Libraries** | PyTorch, Transformers, Sentence-Transformers |
| **Infrastructure** | Docker, Docker Compose, NVIDIA CUDA (GPU), Conda |

---

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/)
- NVIDIA GPU recommended (or ~12 GB RAM for CPU-only mode)

### Quick Start

```bash
git clone <repository-url>
cd FireGPT
docker-compose up
```

Access the UI at **http://localhost:8080**.

### Manual Build

```bash
# Build all images and download models
./build.sh

# Start services
docker-compose up
```

> **CPU-only mode:** Comment out the `deploy` sections in `docker-compose.yml` if running without a GPU.

---

## Usage Examples

| Scenario | What to do |
|:---------|:-----------|
| **Fire Surveillance** | Draw a bounding box over a fire-prone area, place a drone marker, and ask for surveillance waypoints |
| **Fire Suppression** | Mark a fire area and request drone waypoints optimized for water spraying with full area coverage |
| **Search & Rescue** | Upload SAR protocols, mark a search area, and ask for an efficient search pattern prioritizing trails and water sources |
| **Post-Fire Assessment** | Mark a burn area and request a damage assessment flight with hotspot detection passes |
| **Firebreak Planning** | Mark a fire area, place a fire station POI, and ask for a firebreak path to the nearest natural barrier |

---

## Project Structure

```
FireGPT/
├── src/
│   ├── agent/          # LangGraph ReAct agent loop & prompt engineering
│   ├── backend/        # FastAPI server, session management, vision captioning
│   ├── frontend/       # Chat UI, map interface, static assets
│   ├── mcp/            # MCP server with RAG tools & FireGEE client
│   ├── firespread/     # Fire spread modeling components
│   └── ingest/         # PDF/document ingestion into ChromaDB vector stores
├── prompt_supplementary/   # Sample operational documents (SOPs, safety manuals)
├── docker-compose.yml      # Service orchestration
├── Dockerfile.*            # Per-service container definitions
└── build.sh                # Automated build & model download script
```

---

## Three-Tiered Knowledge Base

The RAG system organizes documents into three scoped knowledge bases, queried in priority order:

| Tier | Scope | What it stores | Example documents |
|:-----|:------|:---------------|:------------------|
| **Session** | Per-conversation | Documents uploaded by the user during the active session, specific to the incident at hand | Aerial photos of the current fire, local terrain maps, incident commander notes |
| **Regional** | Country / jurisdiction | Country-specific SOPs and regulations, since different countries have different rules, protocols, and legal requirements for wildfire response | Germany's THW operational guidelines, Australia's CFA suppression procedures, US NWCG standards |
| **Global** | Universal best practices | Fire science reports, post-incident analyses, and cross-border SOPs that are relevant regardless of location | NFPA wildfire standards, historical incident case studies showing what worked and what failed, WHO health advisories for smoke exposure |

The agent always searches **Session first** (most specific), then **Regional**, then **Global** (most general), ensuring that incident-specific context and local regulations take precedence over generic best practices.

---

## How the RAG Pipeline Works

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  1. Document Ingestion      │  PDF/TXT → Token-based chunking (1024 tokens, 128 overlap)
│     via TikToken encoder    │  → BGE-M3 embedding → ChromaDB
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  2. Candidate Retrieval     │  Vector similarity search → Top 50 candidates
│     (Dense Retrieval)       │  Searched in order: Session → Local → Global
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  3. Cross-Encoder Reranking │  BGE-Reranker-V2-M3 scores all 50 candidates
│     (Semantic Precision)    │  → Returns Top 5 with confidence scores
└─────────────┬───────────────┘
              │
              ▼
     Agent Context Window
```

---

## Fire Danger Scoring

Each grid cell in the marked area receives a danger score (0 to 100) based on:

| Factor | Data Source | Contribution |
|:-------|:-----------|:-------------|
| Tree cover density | Hansen Global Forest Change | Up to +30 pts |
| Forest type (coniferous / broadleaf / mixed) | CORINE Land Cover | +5 to +15 pts |
| Recent forest loss (dead fuel) | Hansen Loss Year | Variable |
| Wind speed & direction | NOAA GFS (U/V components) | Variable |
| Temperature | NOAA GFS | Variable |
| Land cover type | CORINE (artificial, wetland, water) | -20 to +0 pts |
| Altitude | SRTM Elevation | Modifies weather effects |

---

## Future Directions

This project was built in **Summer 2025**. Since then, the landscape of open-weight models, agent tooling, and geospatial AI has moved significantly. Below are planned improvements:

### Unified Multimodal Model
The current system runs two separate models: Qwen 3 8B for text reasoning and Qwen 2.5 VL for image analysis. Newer models like **Qwen 3.5** combine strong spatial/image reasoning with text reasoning in a single model. Consolidating to one model would cut VRAM usage, eliminate the separate vision captioning pipeline, and enable the agent to reason over images natively within its tool-use loop rather than through a detached preprocessing step.

### Real-Time Satellite Fire Detection
Replace user-drawn bounding boxes with **live fire hotspot feeds** from NASA FIRMS (VIIRS/MODIS) so the system can auto-detect active fires and trigger assessments without manual input. This would shift FireGPT from a reactive tool to a proactive monitoring system.

### Multi-Agent Orchestration
Split the single ReAct agent into **specialized sub-agents** such as a fire analyst, a logistics planner, and a SAR coordinator, orchestrated via a supervisor pattern. Each agent would own its own MCP tools and memory, enabling parallel reasoning across different aspects of an incident (e.g., assessing fire danger while simultaneously planning evacuation routes).

### Drone API Integration
Connect waypoint generation to real **flight controller APIs** (MAVLink/PX4) to enable direct drone dispatch from the chat interface, closing the loop from analysis to physical action.

### Predictive Fire Spread Modeling
Integrate physics-based or ML-driven **fire propagation models** (e.g., FARSITE, FlamMap, or neural surrogate models) that project fire spread over time using wind, terrain, and fuel data, enabling the agent to reason about where the fire *will be*, not just where it is now.

### Streaming Geospatial Updates
Push **real-time map layer updates** to the frontend via WebSockets as the agent processes GEE data, so users see fire danger heatmaps, POI markers, and drone waypoints appear incrementally rather than waiting for the full response.

---

<div align="center">

Built at **Technical University of Munich** | Applied Machine Intelligence (CIT433042)

</div>
