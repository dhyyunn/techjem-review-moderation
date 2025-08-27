# TikTok TechJem – Review Quality & Relevancy (Starter Repo)

End-to-end pipeline for moderating Google-style location reviews using rules + ML/NLP.
This repo contains code for data prep, feature engineering, multi-label modeling, policy enforcement, evaluation, and a FastAPI demo.

## 0) Setup

```bash
python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # optional, if you plan to use spaCy extras
```

## 1) Data

Place your raw CSV under `data/raw/`. Expected minimal columns:
```
review_id,place_id,place_name,place_category,text,rating,timestamp,user_id,user_review_count,user_avg_rating
```

If you have no labels yet, generate weak labels via rules + relevancy heuristic:
```bash
python -m src.data.preprocess --input data/raw/your.csv --output data/processed/clean.csv --lang en
python -m src.labeling.rule_label --input data/processed/clean.csv --output data/processed/weak_labels.csv
```

## 2) Baseline & Transformer Training

```bash
# Train baseline on weak labels (or your gold labels)
python -m src.modeling.baselines --train data/processed/weak_labels.csv --val_ratio 0.1 --out models/baseline_lr.joblib

# Fine-tune transformer (expects multi-label columns present)
python -m src.modeling.train_multilabel   --train data/processed/weak_labels.csv   --val_ratio 0.1   --model_name distilbert-base-uncased   --labels advertisement_promo irrelevant_content rant_without_visit spam_or_scam low_quality   --out_dir models/distilbert
```

## 3) Evaluation

```bash
python -m src.eval.report --data data/processed/weak_labels.csv --preds models/baseline_lr.joblib --model_type baseline
# or, for transformer checkpoints:
python -m src.eval.report --data data/processed/weak_labels.csv --preds models/distilbert --model_type transformer
```

## 4) Demo API

```bash
uvicorn src.demo.app:app --reload --port 8000
# POST to /classify with JSON: { "text": "...", "place_name": "...", "place_category": "..." }
```

## Notes
- All thresholds are in `src/config/defaults.json` and can be tuned.
- Replace weak labels with your gold labels when available.
- Consider adding cross-encoder relevancy (see `src/modeling/relevancy.py`).

## License
For hackathon/educational use.
