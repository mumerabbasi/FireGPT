from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, UploadFile
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
import uuid
from typing import List, Optional
from typing import Any, Tuple
from fastapi.staticfiles import StaticFiles
import json
import sys
import re
sys.path.append("../agent")
import agent_react
sys.path.append("../ingest")
import ingest_documents as ingest


app = FastAPI()
path_to_frontend = "../frontend"
app.mount("/static", StaticFiles(directory=path_to_frontend), name="static")


def split_think_msg_json(
    raw: str,
    thought_placeholder: str = "",
    json_placeholder: Any = None
) -> Tuple[str, str, Any]:
    """
    Parse out:
      1) the contents of <think>…</think> (or `thought_placeholder` if not found),
      2) the “middle” message text,
      3) the JSON block in json'''…''' (or `json_placeholder` if missing/invalid).
    """

    # 1) Find <think>…</think>
    thought_match = re.search(
        r"<think>\s*(.*?)\s*</think>",
        raw,
        flags=re.DOTALL | re.IGNORECASE
    )
    if thought_match:
        thought = thought_match.group(1).strip()
        message_start = thought_match.end()
    else:
        thought = thought_placeholder
        message_start = 0

    # 2) Find json'''…'''
    json_match = re.search(
        r"json'''(.*?)'''",
        raw,
        flags=re.DOTALL | re.IGNORECASE
    )
    if json_match:
        raw_json = json_match.group(1).strip()
        try:
            json_obj = json.loads(raw_json)
        except json.JSONDecodeError:
            # fallback if it wasn’t valid JSON
            json_obj = json_placeholder
        message_end = json_match.start()
    else:
        json_obj = json_placeholder
        message_end = len(raw)

    # 3) Extract whatever sits between those two markers
    message = raw[message_start:message_end].strip()

    return thought, message, json_obj

@app.get("/")
async def serve_main():
    return FileResponse(path_to_frontend+"/chat_ui.html")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/root/fp/AMI/FireGPT/docs/session"
STORE_DIR = "/root/fp/AMI/FireGPT/stores/session"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STORE_DIR, exist_ok=True)
CHAT_IMAGE_DIR = "chat_images"
os.makedirs(CHAT_IMAGE_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    ingest.build_persistent_store(UPLOAD_DIR, STORE_DIR)
    return {"filename": file.filename, "status": "uploaded"}

@app.get("/list-files")
async def list_files():
    try:
        return os.listdir(UPLOAD_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not list files")

@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "deleted", "filename": filename}
    raise HTTPException(status_code=404, detail="File not found")

@app.delete("/delete-all")
async def delete_all_files():
    deleted = []
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        os.remove(file_path)
        deleted.append(filename)
    return {"status": "all_deleted", "files": deleted}


# @app.post("/map_features")
# async def receive_map_features(request: Request):
#     data = await request.json()  # receive any JSON
#     print("Received JSON data:")
#     print(data)
#     return data  # echo back exactly what was received


@app.on_event("startup")
async def startup_event():
    app.state.graph = await agent_react.compile_graph()
    app.state.thread_id = f"fire-session-{uuid.uuid4()}"



HARDCODED_POIS: List[dict[str, Any]] = [
    {"lat": 47.70, "lon": 7.95, "note": "Critical power sub-station - protect"},
    {"lat": 47.71, "lon": 7.99, "note": "Regional hospital - protect"},
]
HARDCODED_BBOX: Tuple[Tuple[float, float], Tuple[float, float]] = (
    (47.6969, 7.9468),  # top-left  (lat, lon)
    (47.7024, 7.9901),  # bottom-right
)


@app.post("/send-chat")
async def send_chat(
    message: str = Form(...),
    map_features: str = Form(...),
    images: Optional[List[UploadFile]] = File(None)
):
    print(f"Message: {message}")

    # Process uploaded images
    if images:
        for image in images:
            extension = os.path.splitext(image.filename)[1]
            unique_name = f"{uuid.uuid4()}{extension}"
            file_path = os.path.join(CHAT_IMAGE_DIR, unique_name)
            with open(file_path, "wb") as f:
                f.write(await image.read())
            print(f"Saved image: {file_path}")
    else:
        print("No images uploaded.")
    
    # Parse map_features (sent as JSON string)
    try:
        map_features_data = json.loads(map_features)
        print("Parsed map features:", map_features_data)
        bbox_raw = map_features_data["rectangle"][0]["coordinates"]
        bbox = [bbox_raw["nw"], bbox_raw["se"]]
        print(bbox)

    except json.JSONDecodeError:
        map_features_data = None
        print("Invalid map features JSON.")

    # Simulated AI logic and optional path response
    # reply = await agent_react.run_chat(app.state.graph, message, app.state.thread_id)
    sample_message = """
        <think>
        Here is a quick brainstorm for the upcoming sprint:
        - Refactor the parser for better modularity
        - Boost unit-test coverage above 90 %
        - Time-box spike on streaming support
        </think>

        Hey team,

        I just committed the first draft of the revamped parsing module. \nPlease review and leave any comments before noon tomorrow so we can merge by EOD. Please review and leave any comments before noon tomorrow so we can merge by EOD. Please review and leave any comments before noon tomorrow so we can merge by EOD. Please review and leave any comments before noon tomorrow so we can merge by EOD. Please review and leave any comments before noon tomorrow so we can merge by EOD. Please review and leave any comments before noon tomorrow so we can merge by EOD. Please review and leave any comments before noon tomorrow so we can merge by EOD.Please review and leave any comments before noon tomorrow so we can merge by EOD.

        json'''
        [
            {"lat": 47.70, "lng": 7.95},
            {"lat": 47.71, "lng": 7.99},
            {"lat": 47.72, "lng": 8.05}
        ]
        '''
    """
    pathList = [[
        [45.536881434277134, -122.70233760073745],
        [45.51245766875962, -122.6492639957343]
    ]]

    # if map_features_data else []

    # thought, message, wps = split_think_msg_json(reply)
    # print(message)
    # print(wps)
    return {
        "reply": sample_message,
        "pathList": pathList
    }
