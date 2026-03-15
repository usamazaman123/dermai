"""
DermAI — FastAPI Backend
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch, torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import io, numpy as np
from pathlib import Path

# ── App setup ────────────────────────────────────────────────
app = FastAPI(title="DermAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ───────────────────────────────────────────────────
MODEL_PATH  = Path("dermai_efficientnet_b0.pt")
IMG_SIZE    = 224
NUM_CLASSES = 7
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model at startup ────────────────────────────────────
model      = None
checkpoint = None

def load_model():
    global model, checkpoint
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Download from Kaggle output.")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
    in_features = m.num_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    m.load_state_dict(checkpoint["model_state"])
    m.to(DEVICE).eval()
    return m

@app.on_event("startup")
def startup_event():
    global model
    print("Loading DermAI model...")
    model = load_model()
    print(f"Model loaded on {DEVICE}. Ready.")

# ── Preprocessing ────────────────────────────────────────────
def get_transform():
    mean = checkpoint.get("mean", [0.7630, 0.5456, 0.5700])
    std  = checkpoint.get("std",  [0.1409, 0.1526, 0.1697])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

# ── Prediction helper ─────────────────────────────────────────
def predict_image(img: Image.Image):
    tf     = get_transform()
    tensor = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits  = model(tensor)
        probs   = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top3_idx  = probs.argsort()[::-1][:3].tolist()
    idx2label = checkpoint["idx2label"]
    class_names = checkpoint["class_names"]
    severity    = checkpoint["severity"]

    top_label = idx2label[top3_idx[0]]
    results = {
        "prediction": {
            "label":       top_label,
            "name":        class_names[top_label],
            "confidence":  round(float(probs[top3_idx[0]]) * 100, 1),
            "severity":    severity[top_label],
        },
        "top3": [
            {
                "label":      idx2label[i],
                "name":       class_names[idx2label[i]],
                "confidence": round(float(probs[i]) * 100, 1),
            }
            for i in top3_idx
        ],
        "disclaimer": (
            "This tool is for educational purposes only and is NOT a medical diagnosis. "
            "Always consult a qualified dermatologist for any skin concerns."
        ),
    }
    return results

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "DermAI API is running", "device": str(DEVICE)}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/classes")
def get_classes():
    return {
        "classes": checkpoint["class_names"],
        "severity": checkpoint["severity"],
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG).")

    # Read and validate image
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="Image too large. Max 10MB.")

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Upload a valid JPEG or PNG.")

    # Run prediction
    try:
        result = predict_image(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return JSONResponse(content=result)