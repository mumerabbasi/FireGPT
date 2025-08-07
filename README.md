# FireGPT

FireGPT is a comprehensive system for wildfire intelligence and management. It leverages AI agents, geospatial data, and fire spread models to provide real-time insights and decision support for firefighting operations.

## Features

*   **AI-Powered Chat Agent**: Interact with an AI agent to get information about wildfires, operational plans, and safety standards.
*   **Geospatial Analysis**: Utilizes Google Earth Engine for analyzing fire-related geospatial data.
*   **Fire Spread Modeling**: Predicts fire spread using dedicated models.
*   **Document Ingestion**: Ingests and processes supplementary documents like operational plans and safety manuals to provide context to the AI agent.
*   **Web-based UI**: A user-friendly web interface for interacting with the system, including a map view for visualizing fire data.

## Architecture

FireGPT is built on a microservices architecture, with different components containerized using Docker. The services are orchestrated using `docker-compose`.

The main services include:
*   **Backend**: The main server that handles business logic and communication between services.
*   **Frontend**: A web-based user interface.
*   **GEE (Google Earth Engine) Service**: A service for handling Google Earth Engine related tasks.
*   **MCP (Model Context Protocol) Server**: A server for providing model context.
*   **Ollama Service**: Runs large language models for the AI agent.

## Getting Started

### Prerequisites

*   [Docker](https://docs.docker.com/get-docker/)
*   [Docker Compose](https://docs.docker.com/compose/install/)

## Deployment

> This document assumes you have a GPU present in the host machine and necessary docker setup is made for GPUs to work. In case you are running in a GPU-free environment, please comment the "deploy" sections in the `docker-compose.yml` file. Please keep in mind that this means you will load around 12 GB worth of models in RAM.

### Using Docker Compose (Recommended)

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd FireGPT
    ```

2.  **Run the services using Docker Compose:**
    ```bash
    docker-compose up
    ```
    This command will start all the services.

3.  **Access the application:**
    Once the services are running, you can access the web interface by navigating to `http://localhost:8080` (or the appropriate port for the frontend service) in your web browser.

### Manual Build

For developers who want to build the services manually:

1.  **Build the Docker images:**
    You can run the `build.sh` script to build all the necessary Docker images. This script will also download the required models.
    ```bash
    ./build.sh
    ```

2.  **Run the services:**
    Once the images are built, you can start the services using `docker-compose`.
    ```bash
    docker-compose up
    ```

## Usage

After starting the application, you can use the chat interface to ask questions about wildfires. The AI agent will use the available data and models to provide answers. You can also view fire-related information on the map.

## Components

*   `src/backend`: Contains the main backend application logic.
*   `src/frontend`: Contains the HTML, CSS, and JavaScript for the user interface.
*   `src/agent`: Implements the AI agent using a ReAct-style loop.
*   `src/firespread`: Contains the fire spread modeling components.
*   `src/ingest`: Handles the ingestion of PDF documents into a vector store.
*   `src/mcp`: The Model Context Protocol server.
*   `docker-compose.yml`: Defines and configures all the services.
*   `Dockerfile.*`: Dockerfiles for building the individual services.
*   `huggingface_models`: Contains the downloaded Hugging Face models for embedding and reranking.
*   `prompt_supplementary`: Contains supplementary documents used for providing context to the agent.
