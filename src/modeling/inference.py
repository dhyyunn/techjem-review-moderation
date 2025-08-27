import argparse, pandas as pd, numpy as np, joblib, torch, json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def infer_baseline(model_path, texts):
    pack = joblib.load(model_path)
    pipe = pack["pipeline"]
    labels = pack["labels"]
    probs = pipe.predict_proba(texts)
    if isinstance(probs, list):  # list of arrays per class from OneVsRest
        probs = np.vstack([p[:,1] if p.ndim==2 else p for p in probs]).T
    return labels, probs

def infer_transformer(model_dir, texts, thresh=0.5):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits.numpy()
    probs = 1/(1+np.exp(-logits))
    labels = json.load(open(f"{model_dir}/config.json"))["id2label"]
    labels = [labels[str(i)] for i in range(len(labels))] if labels else None
    return labels, probs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to joblib (baseline) or dir (transformer).")
    ap.add_argument("--model_type", choices=["baseline","transformer"], required=True)
    ap.add_argument("--text", nargs="+", required=True)
    args = ap.parse_args()

    if args.model_type == "baseline":
        labels, probs = infer_baseline(args.model, args.text)
    else:
        labels, probs = infer_transformer(args.model, args.text)

    for i, t in enumerate(args.text):
        row = {labels[j]: float(probs[i,j]) for j in range(len(labels))}
        print(f"\nTEXT: {t}\nPROBS: {row}")
