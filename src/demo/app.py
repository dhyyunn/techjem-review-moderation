from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import os, json, joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np

# Load baseline by default if exists; otherwise lazy-load transformer if configured
BASELINE_PATH = os.environ.get("BASELINE_PATH", "models/baseline_lr.joblib")
TRANSFORMER_DIR = os.environ.get("TRANSFORMER_DIR", "models/distilbert")

app = FastAPI(title="Review Moderation Demo")

class ReviewIn(BaseModel):
    text: str
    place_name: str = ""
    place_category: str = ""

def infer_baseline(text: str):
    pack = joblib.load(BASELINE_PATH)
    pipe = pack["pipeline"]
    labels = pack["labels"]
    prob = np.array([p[0,1] if p.ndim==2 else p[0] for p in pipe.predict_proba([text])]).reshape(1,-1)
    return labels, prob[0].tolist()

def infer_transformer(text: str):
    tok = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
    enc = tok([text], padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits.numpy()
    prob = 1/(1+np.exp(-logits))
    cfg = json.load(open(f"{TRANSFORMER_DIR}/config.json"))
    labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
    return labels, prob[0].tolist()

@app.post("/classify")
def classify(inp: ReviewIn):
    text = inp.text.strip()
    if os.path.exists(BASELINE_PATH):
        labels, probs = infer_baseline(text)
    elif os.path.exists(TRANSFORMER_DIR):
        labels, probs = infer_transformer(text)
    else:
        return {"error":"No model found. Train baseline or transformer first."}

    out = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return {"probs": out, "threshold": 0.5}
