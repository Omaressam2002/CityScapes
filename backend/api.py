from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from PIL import Image
import torch
import numpy as np
import io
import os
from pydantic import BaseModel
from Omar import load_model, DEVICE, Inference

app = FastAPI()
model = load_model()

# ---- Request Schema ----
class PathRequest(BaseModel):
    image_path: str

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/predict")
def predict(data: PathRequest):

    input_path = data.image_path

    if not os.path.exists(input_path):
        return {"error": "File not found"}

    # ---- Your preprocessing ----
    ret = Inference(model, input_path)
    print(f"Mask Path: {ret['mask']}\nColorful Mask Path: {ret['colorful']}")

    return {"mask_path": ret["mask"]}

