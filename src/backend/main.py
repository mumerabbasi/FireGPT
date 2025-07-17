"""FastAPI application providing a simple chat and file-upload API.
"""

from __future__ import annotations

import os
import json
import re
import ast
import sys
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple
import shutil
import base64

from fastapi import File, Form, HTTPException, UploadFile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------------
# Third-party / local imports (paths added dynamically)
# ---------------------------------------------------------------------------

sys.path.append("../agent")
import agent_react  # type: ignore  # noqa: E402

sys.path.append("../ingest")
import ingest_documents as ingest  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------
# Directories & configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("/root/fp/AMI/FireGPT")
UPLOAD_DIR = BASE_DIR / "docs/session"
STORE_DIR = BASE_DIR / "stores/session"
CHAT_IMAGE_DIR = BASE_DIR / "chat_images"
SESSION_DIRS = (UPLOAD_DIR, STORE_DIR, CHAT_IMAGE_DIR)
FRONTEND_DIR = Path("../frontend")

# Vision Language Model API Configuration
OPENAI_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
OPENAI_MODEL = os.getenv("FGPT_MODEL", "qwen2.5vl")
# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI()
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_FENCE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
_FIRST_BRACE_LIST = re.compile(r"(\{.*?\}|\[.*?\])", re.DOTALL)
_COMMENT_LINE = re.compile(r"//.*?(?=\n|$)")
_TRAILING_COMMA = re.compile(r",(\s*[\]}])")


def _strip_comments_and_commas(txt: str) -> str:
    """Remove //-style comments and any comma that comes *right* before ] or }."""
    txt = _COMMENT_LINE.sub("", txt)
    txt = _TRAILING_COMMA.sub(r"\1", txt)
    return txt


def parse_agent_reply(raw: str) -> Tuple[str, Optional[Any]]:
    """
    Return (message_without_thinks, parsed_json_or_None).
    """
    # 1Drop every <think>…</think>
    message = _THINK_BLOCK.sub("", raw).strip()

    # Locate JSON text
    m = _JSON_FENCE.search(message)
    json_text = m.group(1) if m else None
    if json_text is None:
        m2 = _FIRST_BRACE_LIST.search(message)
        json_text = m2.group(1) if m2 else None

    # Parse, tolerating comments & trailing commas
    data: Optional[Any] = None
    if json_text:
        cleaned = _strip_comments_and_commas(json_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(cleaned)
            except Exception:
                pass

    return message, data


def parse_map_features(raw: str) -> tuple[Optional[dict[str, Any]], Optional[list[dict[str, Any]]]]:
    """Convert *map_features* JSON into ``fire_bbox`` and ``pois`` structures.

    Example map_features:
    ```json
    {
        "rectangle": [
            {
                "name": "AREA",
                "coordinates": {
                    "nw": {"lat": 1, "lng": 2},
                    "se": {"lat": 3, "lng": 4}
                },
                "description": "…"
            }
        ],
        "marker": [ {"name": "FIRE", …} ]
    }
    ```
    Returns ``(fire_bbox, pois)`` where *fire_bbox* is ``None`` if no
    suitable rectangle is found and *pois* is ``None`` if no markers are
    provided.
    """

    try:
        features = json.loads(raw)
    except json.JSONDecodeError:
        return None, None

    # Extract AREA rectangle -> fire_bbox
    fire_bbox: Optional[dict[str, Any]] = None
    for rect in features.get("rectangle", []):
        if rect.get("name", "").upper() == "AREA":
            coords = rect.get("coordinates", {})
            fire_bbox = {
                "top_left": coords.get("nw"),
                "bottom_right": coords.get("se"),
                "description": rect.get("description", ""),
            }
            break

    pois: Optional[List[dict[str, Any]]] = features.get("marker") or None

    return fire_bbox, pois


async def caption_images(
    llm,
    images: List[UploadFile],
    save_dir: Path = CHAT_IMAGE_DIR,
) -> Tuple[List[str], List[Path]]:
    """
    Save each uploaded image, ask the multimodal LLM for an incident-focused
    description, and return (captions, saved_paths).
    """
    captions: List[str] = []
    saved_paths: List[Path] = []

    # Save the images to the chat image directory
    for img in images:
        fname = f"{uuid.uuid4()}{Path(img.filename).suffix}"
        path = CHAT_IMAGE_DIR / fname
        path.write_bytes(await img.read())
        saved_paths.append(path)

    if saved_paths:
        for p in saved_paths:
            # Encode as data URL for the OpenAI-compat endpoint
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            data_url = (
                f"data:{mime};base64," +
                base64.b64encode(p.read_bytes()).decode("utf-8")
            )

            vision_prompt = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "You are an image analyst for a wildfire-incident assistant.\n"
                            "Describe all the elements you see in this image (e.g. fire "
                            "perimeter, attack routes, water pick-up points, helicopter "
                            "lanes, labelled flanks, legend items et) in detail."
                            "List them as detailed bullet points."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            )

            # Call the multimodal model asynchronously
            ai_msg = await app.state.llm.ainvoke([vision_prompt])
            captions.append(ai_msg.content.strip())

    return captions, saved_paths


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_main() -> FileResponse:
    """Serve the front-end chat UI."""

    return FileResponse(FRONTEND_DIR / "chat_ui.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> dict[str, str]:
    """Upload a document then rebuild the vector store."""

    destination = UPLOAD_DIR / file.filename
    destination.write_bytes(await file.read())

    ingest.build_session_store(str(UPLOAD_DIR))
    # ingest.build_session_store([destination])
    return {"filename": file.filename, "status": "uploaded"}


@app.get("/list-files")
async def list_files() -> List[str]:
    """Return all filenames currently stored in *UPLOAD_DIR*."""

    try:
        return [p.name for p in UPLOAD_DIR.iterdir() if p.is_file()]
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not list files") from exc


# TODO: implement deleting collection from the vector store
@app.delete("/delete/{filename}")
async def delete_file(filename: str) -> dict[str, str]:
    """Delete a single upload by *filename*."""

    try:
        (UPLOAD_DIR / filename).unlink(missing_ok=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc

    return {"status": "deleted", "filename": filename}


@app.delete("/delete-all")
async def delete_all_files():
    deleted = []
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        os.remove(file_path)
        deleted.append(filename)

    # Delete all session directories
    for d in SESSION_DIRS:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)  # fresh, empty dir

    return {"status": "all_deleted", "files": deleted}


@app.post("/send-chat")
async def send_chat(
    message: str = Form(...),
    map_features: str = Form(...),
    images: Optional[List[UploadFile]] = File(None),
) -> dict[str, Any]:
    """Main chat endpoint - handles text, map geometry, and inline images."""

    # Generate caption for the chat message
    captions: List[str] = []
    if images:
        captions, _ = await caption_images(app.state.llm, images)

    full_prompt = message
    if captions:
        full_prompt += "\n\nANALYSIS OF USER PROVIDED IMAGES:\n" + "\n".join(f"- {c}" for c in captions)

    # Convert map_features to fire_bbox + pois
    fire_bbox, pois = parse_map_features(map_features)

    # Call LLM agent
    raw_reply = await agent_react.run_chat(
        graph=app.state.graph,
        thread_id=app.state.thread_id,
        user_prompt=full_prompt,
        fire_bbox=fire_bbox,
        pois=pois,
    )
    clean_msg, path_list = parse_agent_reply(raw_reply)

    return {
        "reply": clean_msg,
        "pathList": [path_list] or [],  # defaults to empty list
        # "reply": full_prompt,
        # "pathList": [],
    }

# ---------------------------------------------------------------------------
# Lifespan events
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def _startup() -> None:  # noqa: D401
    """Boot the agent."""
    # Initialize the agent graph and thread ID
    app.state.graph = await agent_react.compile_graph()
    app.state.thread_id = f"fire-session-{uuid.uuid4()}"

    # Initialize the Vision API client
    app.state.llm = ChatOpenAI(
        model_name=OPENAI_MODEL,
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_base=OPENAI_BASE,
        openai_api_key="unused",
        model_kwargs={"tool_choice": "any"},
    )


@app.on_event("shutdown")
async def shutdown() -> None:  # noqa: D401
    """Reset session state and clean up directories."""
    # Ensure all session directories exist and are empty
    for d in SESSION_DIRS:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)  # fresh, empty dir
