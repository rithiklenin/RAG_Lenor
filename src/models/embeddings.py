import requests
from typing import List
from llama_index.legacy.embeddings.huggingface import HuggingFaceEmbedding
from ..config.settings import HUGGINGFACE_API_KEY, HF_EMBEDDER_API_URL, DEFAULT_EMBEDDING_MODEL

def get_dense_embeddings(payload: List[str]) -> List[float]:
    """Get dense embeddings using HuggingFace API."""
    response = requests.post(
        HF_EMBEDDER_API_URL,
        headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
        json=payload
    )
    return response.json()

def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load the HuggingFace embedding model."""
    return HuggingFaceEmbedding(model_name) 