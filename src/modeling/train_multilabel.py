import argparse, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from typing import List, Dict

def make_dataset(df, labels: List[str], tok, max_len=256):
    texts = df["text_norm"].fillna("").tolist()
    enc = tok(texts, padding=True, truncation=True, max_length=max_len)
    y = df[labels].astype(float).values
    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(texts)
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k,v in enc.items()}
            item["labels"] = torch.tensor(y[i])
            return item
    return DS()

def compute_metrics_builder(labels: List[str], thresh=0.5):
    from sklearn.metrics import f1_score, precision_score, recall_score
    def compute(eval_pred):
        logits, y_true = eval_pred
        y_prob = 1/(1+np.exp(-logits))
        y_pred = (y_prob >= thresh).astype(int)
        return {
            "micro/f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "macro/f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "micro/precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "micro/recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        }
    return compute

def main(train_path, out_dir, model_name, labels, val_ratio=0.1, epochs=3, lr=2e-5, seed=42):
    df = pd.read_csv(train_path)
    for l in labels:
        assert l in df.columns, f"Missing label column: {l}"

    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed, stratify=df[labels].astype(int).values.sum(axis=1) > 0)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels), problem_type="multi_label_classification")

    ds_train = make_dataset(train_df, labels, tok)
    ds_val = make_dataset(val_df, labels, tok)

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro/f1",
        logging_steps=50,
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        compute_metrics=compute_metrics_builder(labels)
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Saved transformer to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()
    main(args.train, args.out_dir, args.model_name, args.labels, args.val_ratio, args.epochs, args.lr)
