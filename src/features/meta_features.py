import pandas as pd, numpy as np

def meta_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "rating": pd.to_numeric(df.get("rating", np.nan), errors="coerce"),
        "user_review_count": pd.to_numeric(df.get("user_review_count", np.nan), errors="coerce"),
        "user_avg_rating": pd.to_numeric(df.get("user_avg_rating", np.nan), errors="coerce"),
    })
    # Temporal signals if timestamp present
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        out["hour"] = ts.dt.hour
        out["dow"] = ts.dt.dayofweek
    else:
        out["hour"] = np.nan
        out["dow"] = np.nan
    return out.fillna(0.0)
