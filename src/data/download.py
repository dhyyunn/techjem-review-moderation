"""
Download/ingest helpers. For Kaggle or manual files, place the CSV in data/raw and call preprocess.
"""
import argparse, os, pandas as pd

def read_local_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Put your CSV under data/raw/." )
    return pd.read_csv(path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to a local raw CSV file.")
    args = ap.parse_args()
    df = read_local_csv(args.path)
    print(df.head(3))
