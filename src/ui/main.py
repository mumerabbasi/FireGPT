from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, UploadFile
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
from typing import List, Optional

import os
from os.path import dirname, realpath, sep, pardir
import sys
sys.path.append("../agent")
import agent_react
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)
CHAT_IMAGE_DIR = "chat_images"
os.makedirs(CHAT_IMAGE_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

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


@app.post("/map_features")
async def receive_map_features(request: Request):
    data = await request.json()  # receive any JSON
    print("Received JSON data:")
    print(data)
    return data  # echo back exactly what was received


@app.post("/send-chat")
async def send_chat(
    message: str = Form(...),
    images: Optional[List[UploadFile]] = File(None)
):    
    saved_image_urls = []
    print(f"Message: {message}")
    if images:
        for image in images:
            extension = os.path.splitext(image.filename)[1]
            unique_name = f"{uuid.uuid4()}{extension}"
            file_path = os.path.join(CHAT_IMAGE_DIR, unique_name)
            with open(file_path, "wb") as f:
                f.write(await image.read())
    else:
        print("No images uploaded.")


    # Simulated response (replace with AI logic)
    reply = await agent_react.run_chat(message)
    # reply = f"I received your message: '{reply}'"
    return {
        "reply": reply
    }

    # Mock assistant reply
    return {"response": "Thanks! I received your message."}