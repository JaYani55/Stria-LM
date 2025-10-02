from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np
import openai
from typing import Dict, Any
from .config import EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL

@lru_cache(maxsize=4)
def get_local_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Loads a local sentence-transformer model and caches it.
    """
    print(f"Loading local embedding model: {model_name}")
    return SentenceTransformer(model_name)

def _generate_local_embedding(text: str, model_config: Dict[str, Any]) -> np.ndarray:
    """
    Generates a vector embedding using a local SentenceTransformer model.
    """
    model = get_local_embedding_model(model_config["model"])
    embedding = model.encode(text)
    return embedding.astype(np.float32)

def _generate_openai_embedding(text: str, model_config: Dict[str, Any]) -> np.ndarray:
    """
    Generates a vector embedding using an OpenAI-compliant API.
    """
    client = openai.OpenAI(
        base_url=model_config.get("base_url"),
        api_key=model_config.get("api_key"),
    )
    response = client.embeddings.create(
        input=[text],
        model=model_config["model"]
    )
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)

def generate_embedding(text: str, model_id: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """
    Generates a vector embedding for the given text based on the model configuration.
    It can look up models by their ID (e.g., 'local_default') or by the model name
    (e.g., 'sentence-transformers/all-MiniLM-L6-v2') as a fallback.
    """
    # First, try to find the model by its configuration ID
    model_config = EMBEDDING_MODELS.get(model_id)

    # If not found, search for a configuration where the 'model' field matches the model_id
    if not model_config:
        for config in EMBEDDING_MODELS.values():
            if config.get("model") == model_id:
                model_config = config
                break

    if not model_config:
        raise ValueError(f"Embedding model '{model_id}' not found in configuration.")

    category = model_config.get("category")
    
    if category == "Local":
        return _generate_local_embedding(text, model_config)
    elif category == "OpenAI API":
        return _generate_openai_embedding(text, model_config)
    else:
        raise NotImplementedError(f"Embedding category '{category}' is not supported.")

def get_vector_dimension(model_id: str = DEFAULT_EMBEDDING_MODEL) -> int:
    """
    Determines the vector dimension for a given model ID.
    """
    # First, try to find the model by its configuration ID
    model_config = EMBEDDING_MODELS.get(model_id)

    # If not found, search for a configuration where the 'model' field matches the model_id
    if not model_config:
        for config in EMBEDDING_MODELS.values():
            if config.get("model") == model_id:
                model_config = config
                break

    if not model_config:
        raise ValueError(f"Embedding model '{model_id}' not found in configuration.")

    category = model_config.get("category")
    
    if category == "Local":
        model = get_local_embedding_model(model_config["model"])
        return model.get_sentence_embedding_dimension()
    elif category == "OpenAI API":
        # For OpenAI models, we generate a dummy embedding to find the dimension.
        # This is a bit wasteful but happens only at project creation.
        return len(_generate_openai_embedding("dimension check", model_config))
    else:
        raise NotImplementedError(f"Cannot determine dimension for category '{category}'.")
