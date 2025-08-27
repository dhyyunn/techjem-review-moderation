import argparse, pandas as pd, numpy as np
from tqdm import tqdm
from src.features.text_features import TextFeatureExtractor, place_prompt
from src.policy.rules import rule_advertisement, rule_rant_no_visit, rule_spam, rule_low_quality, rule_irrelevant_from_relevancy

LABELS = ["advertisement_promo","irrelevant_content","rant_without_visit","spam_or_scam","low_quality"]

def main(inp, out):
    df = pd.read_csv(inp)
    tfe = TextFeatureExtractor()
    prompts = [place_prompt(n, c) for n, c in zip(df.get("place_name",""), df.get("place_category",""))]
    rel_scores = tfe.relevancy_score(df["text_norm"].tolist(), prompts)

    labels = []
    for text, rel in tqdm(zip(df["text_norm"], rel_scores), total=len(df)):
        row = {
            "advertisement_promo": int(rule_advertisement(text)),
            "irrelevant_content": int(rule_irrelevant_from_relevancy(rel, tau=0.25)),
            "rant_without_visit": int(rule_rant_no_visit(text)),
            "spam_or_scam": int(rule_spam(text)),
            "low_quality": int(rule_low_quality(text)),
        }
        labels.append(row)

    lab_df = pd.DataFrame(labels)
    out_df = pd.concat([df.reset_index(drop=True), lab_df], axis=1)
    out_df.to_csv(out, index=False)
    print(f"Wrote weak labels to {out} with shape {out_df.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    main(args.input, args.output)
