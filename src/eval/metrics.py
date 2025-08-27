import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro = {
        "macro_precision": prec.mean(),
        "macro_recall": rec.mean(),
        "macro_f1": f1.mean()
    }
    per_label = {f"{labels[i]}_{k}": v for i in range(len(labels)) for k,v in zip(["precision","recall","f1"], [prec[i], rec[i], f1[i]])}
    try:
        aucs = roc_auc_score(y_true, y_prob, average=None)
        per_auc = {f"{labels[i]}_auc": aucs[i] for i in range(len(labels))}
    except Exception:
        per_auc = {}
    return {**macro, **per_label, **per_auc}
