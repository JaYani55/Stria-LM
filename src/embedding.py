from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=4)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Loads a sentence-transformer model and caches it.
    """
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)

def generate_embedding(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generates a vector embedding for the given text and ensures it's a float32 numpy array.
    """
    model = get_embedding_model(model_name)
    embedding = model.encode(text)
    return embedding.astype(np.float32)
