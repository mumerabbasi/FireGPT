from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os

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
