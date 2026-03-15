# DermAI — Skin Cancer Detection System

End-to-end skin cancer classifier using EfficientNet-B0 trained on HAM10000.

---

## Project Structure

```
dermai/
├── kaggle_training.py     ← Run this on Kaggle to train the model
├── frontend/
│   └── index.html         ← Open in browser (no build step needed)
└── backend/
    ├── main.py            ← FastAPI server
    ├── requirements.txt   ← Python dependencies
    └── dermai_efficientnet_b0.pt  ← (place here after Kaggle training)
```

---

## Step 1 — Train model on Kaggle

1. Go to https://www.kaggle.com and create a new notebook
2. Add dataset: search "skin-cancer-mnist-ham10000" → Add data
3. Enable GPU: Settings → Accelerator → GPU P100
4. Copy-paste the entire content of `kaggle_training.py` into the notebook
5. Run all cells (~45 minutes)
6. Download `dermai_efficientnet_b0.pt` from the Output tab
7. Place the `.pt` file inside the `backend/` folder

---

## Step 2 — Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Test it: http://localhost:8000/health

---

## Step 3 — Open the frontend

Just open `frontend/index.html` in your browser.
No build step, no npm — pure HTML/CSS/JS.

---

## 7 Disease Classes

| Code  | Disease                | Severity |
|-------|------------------------|----------|
| mel   | Melanoma               | HIGH     |
| bcc   | Basal Cell Carcinoma   | HIGH     |
| akiec | Actinic Keratosis      | MEDIUM   |
| bkl   | Benign Keratosis       | LOW      |
| df    | Dermatofibroma         | LOW      |
| nv    | Melanocytic Nevus      | LOW      |
| vasc  | Vascular Lesion        | LOW      |

---

## Expected Results

- Validation accuracy: 85–92%
- Inference time: <500ms on CPU, <100ms on GPU
- Model file size: ~20MB

---

## Disclaimer

This system is for educational/research purposes only.
It is NOT a substitute for professional medical diagnosis.
Always consult a qualified dermatologist.
