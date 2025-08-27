import argparse, os, re, pandas as pd, numpy as np
from langdetect import detect, DetectorFactory
from cleantext import clean
from sklearn.model_selection import train_test_split

DetectorFactory.seed = 42

def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    # Lowercase & clean (preserve emojis)
    s = s.strip().lower()
    s = clean(s,
              fix_unicode=True,
              to_ascii=False,
              lower=True,
              no_urls=False,   # keep URLs for ad detection; rules handle them
              no_emails=False,
              no_phone_numbers=False,
              no_numbers=False,
              no_digits=False,
              no_currency_symbols=False,
              no_punct=False)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def detect_lang_safe(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

def preprocess(df: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    needed = ["review_id","place_id","place_name","place_category","text","rating","timestamp","user_id","user_review_count","user_avg_rating"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    df["text"] = df["text"].astype(str).fillna("")
    df["text_norm"] = df["text"].apply(normalize_text)
    df["lang"] = df["text_norm"].apply(detect_lang_safe)

    if lang:
        df = df[df["lang"] == lang]

    # Drop empty or very short
    df = df[df["text_norm"].str.split().str.len() >= 2].copy()

    # Simple duplicate removal (exact text duplicate per place)
    df = df.drop_duplicates(subset=["place_id", "text_norm"])

    # Basic type conversions
    for c in ["rating","user_review_count","user_avg_rating"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    raw = pd.read_csv(args.input)
    clean_df = preprocess(raw, lang=args.lang)
    clean_df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(clean_df)} rows")
