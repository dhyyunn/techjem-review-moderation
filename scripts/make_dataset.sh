#!/usr/bin/env bash
set -euo pipefail
python -m src.data.preprocess --input data/raw/input.csv --output data/processed/clean.csv --lang en
python -m src.labeling.rule_label --input data/processed/clean.csv --output data/processed/weak_labels.csv
echo "Wrote data/processed/weak_labels.csv"
