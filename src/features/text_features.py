from typing import Dict, List
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer, util

class TextFeatureExtractor:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    @staticmethod
    def basic_signals(texts: List[str]) -> pd.DataFrame:
        import re, emoji
        rows = []
        url_re = re.compile(r"(https?://|www\.)", re.I)
        phone_re = re.compile(r"\+?\d[\d\-\s]{6,}\d")
        for t in texts:
            words = t.split()
            rows.append({
                "n_chars": len(t),
                "n_words": len(words),
                "has_url": int(bool(url_re.search(t))),
                "has_phone": int(bool(phone_re.search(t))),
                "emoji_cnt": sum(1 for ch in t if ch in emoji.EMOJI_DATA),
                "caps_ratio": sum(c.isupper() for c in t) / max(1, len(t)),
                "punct_ratio": sum(c in '.,;:!?()[]{}' for c in t) / max(1, len(t)),
            })
        return pd.DataFrame(rows)

    def relevancy_score(self, review_texts: List[str], place_prompts: List[str]) -> np.ndarray:
        a = self.embed(review_texts)
        b = self.embed(place_prompts)
        sims = util.cos_sim(a, b).diagonal()
        return np.array(sims).astype(float)

def place_prompt(name: str, category: str) -> str:
    cat = category or "place"
    return f"{name}. Category: {cat}."
