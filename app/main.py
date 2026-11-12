import requests
import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()


# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image_array(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    resized = cv2.resize(thresh, None, fx=1.5, fy=1.5)

    return resized


# -----------------------------
# Encode Image
# -----------------------------
def encode_image(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode()


# -----------------------------
# LLM Extraction (Ollama)
# -----------------------------
def extract_drugs_with_ollama(image_base64):
    prompt = """
You are a medical assistant.

Extract ONLY drug names from this prescription image.

Strict rules:
- Do NOT guess
- Do NOT add drugs not visible
- Ignore unclear words
- Ignore numbers and dosage
- Return ONLY a python list

Example:
["Paracetamol", "Augmentin"]
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-ocr",
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }
    )

    data = response.json()

    # safe handling
    if "response" in data:
        return data["response"]
    elif "message" in data:
        return data["message"]["content"]
    else:
        return str(data)


# -----------------------------
# API Endpoint
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Medical Prescription Drug Extractor API"}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # convert to image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # preprocess
        processed = preprocess_image_array(img)

        # encode
        base64_img = encode_image(processed)

        # extract drugs
        result = extract_drugs_with_ollama(base64_img)

        return JSONResponse(content={
            "drugs": result
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )