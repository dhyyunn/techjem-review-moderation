#!/usr/bin/env bash
set -euo pipefail
python -m src.eval.report --data data/processed/weak_labels.csv --preds models/baseline_lr.joblib --model_type baseline
python -m src.eval.report --data data/processed/weak_labels.csv --preds models/distilbert --model_type transformer
