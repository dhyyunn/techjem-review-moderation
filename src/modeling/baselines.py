import argparse, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

DEFAULT_LABELS = ["advertisement_promo","irrelevant_content","rant_without_visit","spam_or_scam","low_quality"]

def train_baseline(train_path, out_path, labels=None, val_ratio=0.1, seed=42):
    df = pd.read_csv(train_path)
    labels = labels or [c for c in DEFAULT_LABELS if c in df.columns]
    assert labels, "No label columns found!"

    X_train, X_val, y_train, y_val = train_test_split(
        df["text_norm"].fillna(""), df[labels].astype(int).values, test_size=val_ratio, random_state=seed, stratify=df[labels].astype(int).values.sum(axis=1) > 0
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=200, n_jobs=None)))
    ])
    pipe.fit(X_train, y_train)
    preds_val = (pipe.predict_proba(X_val) >= 0.5).astype(int)

    print("Validation report:")
    print(classification_report(y_val, preds_val, target_names=labels, zero_division=0))

    joblib.dump({"pipeline": pipe, "labels": labels}, out_path)
    print(f"Saved baseline model to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--labels", nargs="*", default=None)
    args = ap.parse_args()
    train_baseline(args.train, args.out, labels=args.labels, val_ratio=args.val_ratio)
