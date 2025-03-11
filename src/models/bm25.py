import os
import pickle
from pinecone_text.sparse import BM25Encoder

class BM25Singleton:
    _instance = None

    @classmethod
    def get_instance(cls, texts=None):
        if cls._instance is None:
            if texts is None:
                raise ValueError("Initial texts required for the first initialization!")
            cls._instance = cls(texts)
        return cls._instance

    def __init__(self):
        self.bm25 = BM25Encoder()

    def fit(self, texts):
        self.bm25.fit(texts)

    def encode(self, queries):
        return self.bm25.encode_documents(queries)

def save_bm25_instance(model_instance, model_path):
    """Save BM25 model to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(model_instance, file)

def load_bm25_instance(pickle_path):
    """Load BM25 model from disk."""
    with open(pickle_path, "rb") as file:
        return pickle.load(file) 