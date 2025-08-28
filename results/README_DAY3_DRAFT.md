# TechJem Review Moderation — Day 3 Draft
Generated: 2025-08-28 06:34
## What’s included
- Day-2 model with metadata features
- Batch predictions CSV for demo (in Drive)
- Gradio interactive demo link (from notebook)

## Artifacts (in Drive)
- Model: /content/drive/MyDrive/TechJemShared/models/pipeline_day2.joblib
- Metrics: /content/drive/MyDrive/TechJemShared/results/metrics_day2_baseline.json
- Confusion matrix: /content/drive/MyDrive/TechJemShared/results/cm_day2_baseline.png
- Thresholded metrics: /content/drive/MyDrive/TechJemShared/results/metrics_day2_thresholded.json
- Demo predictions: /content/drive/MyDrive/TechJemShared/results/demo_predictions_day3.csv

## Key metrics
- Baseline macro-F1: 0.4921219509909133
- Thresholded macro-F1: 0.528132753494206

## Notes
- Labels: VALID / ADVERTISEMENT / IRRELEVANT / RANT_NO_VISIT
- Model: TF-IDF (1–2 n-grams) + metadata → Logistic Regression (balanced)
- Next: record demo video, polish README, merge PR to main.
