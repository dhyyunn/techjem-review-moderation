import argparse, pandas as pd, numpy as np, joblib, os, json, torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.eval.metrics import multilabel_metrics

def eval_baseline(data_path, model_path, labels=None, val_ratio=0.1, seed=42):
    df = pd.read_csv(data_path)
    labels = labels or joblib.load(model_path)["labels"]
    df_tr, df_val = train_test_split(df, test_size=val_ratio, random_state=seed, stratify=df[labels].astype(int).values.sum(axis=1) > 0)

    pack = joblib.load(model_path)
    pipe = pack["pipeline"]
    y_true = df_val[labels].astype(int).values
    y_prob = pipe.predict_proba(df_val["text_norm"].fillna(""))
    if isinstance(y_prob, list):
        y_prob = np.vstack([p[:,1] if p.ndim==2 else p for p in y_prob]).T
    return multilabel_metrics(y_true, y_prob, labels)

def eval_transformer(data_path, model_dir, labels=None, val_ratio=0.1, seed=42):
    df = pd.read_csv(data_path)
    if labels is None:
        cfg_path = os.path.join(model_dir, "config.json")
        if os.path.exists(cfg_path):
            id2label = json.load(open(cfg_path)).get("id2label", {})
            labels = [id2label[str(i)] for i in range(len(id2label))]
    df_tr, df_val = train_test_split(df, test_size=val_ratio, random_state=seed, stratify=df[labels].astype(int).values.sum(axis=1) > 0)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    enc = tok(df_val["text_norm"].fillna("").tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits.numpy()
    y_prob = 1/(1+np.exp(-logits))
    y_true = df_val[labels].astype(int).values
    return multilabel_metrics(y_true, y_prob, labels)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True, help="joblib (baseline) or model dir (transformer)")
    ap.add_argument("--model_type", choices=["baseline","transformer"], required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()

    if args.model_type == "baseline":
        out = eval_baseline(args.data, args.preds, val_ratio=args.val_ratio)
    else:
        out = eval_transformer(args.data, args.preds, val_ratio=args.val_ratio)
    print(out)
