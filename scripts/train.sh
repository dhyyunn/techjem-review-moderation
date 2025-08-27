#!/usr/bin/env bash
set -euo pipefail
python -m src.modeling.baselines --train data/processed/weak_labels.csv --val_ratio 0.1 --out models/baseline_lr.joblib
python -m src.modeling.train_multilabel   --train data/processed/weak_labels.csv   --val_ratio 0.1   --model_name distilbert-base-uncased   --labels advertisement_promo irrelevant_content rant_without_visit spam_or_scam low_quality   --out_dir models/distilbert
