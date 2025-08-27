from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer, util

class BiEncoderRelevancy:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def score(self, reviews: List[str], place_prompts: List[str]) -> np.ndarray:
        a = self.model.encode(reviews, normalize_embeddings=True, convert_to_numpy=True)
        b = self.model.encode(place_prompts, normalize_embeddings=True, convert_to_numpy=True)
        sims = util.cos_sim(a, b).diagonal()
        return np.array(sims).astype(float)
