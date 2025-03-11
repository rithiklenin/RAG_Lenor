import logging
import time
from typing import List, Dict, Any
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from ..config.settings import PINECONE_API_KEY

class PineconeService:
    def __init__(self, index_name: str):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = index_name
        self.index = None
        self._initialize_index()

    def _initialize_index(self):
        """Initialize Pinecone index."""
        if self.index_name not in self.pc.list_indexes().names():
            logging.info("Creating pinecone index...")
            self.pc.create_index(
                self.index_name,
                dimension=768,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(self.index_name)

    def upsert_vectors(self, vectors: List[Dict[str, Any]], namespace: str, batch_size: int = 10):
        """Upsert vectors to Pinecone index."""
        logging.info(f"Starting upsertion to namespace {namespace}...")
        total_vectors = len(vectors)
        
        for batch_start in tqdm(range(0, total_vectors, batch_size), desc="Processing and Upserting Batches"):
            batch_end = min(batch_start + batch_size, total_vectors)
            batch = vectors[batch_start:batch_end]
            
            try:
                self.index.upsert(vectors=batch, namespace=namespace)
                logging.info(f"Upserted batch {batch_start // batch_size + 1}...")
            except Exception as e:
                logging.error(f"Error upserting batch: {e}")
                return False

        time.sleep(10)
        index_status = self.index.describe_index_stats()
        
        if index_status["namespaces"][namespace]["vector_count"] == total_vectors:
            logging.info(f"All vectors uploaded successfully to namespace {namespace}")
            return True
        else:
            logging.error(f"Not all vectors were upserted to namespace {namespace}")
            return False

    def query(self, namespace: str, query_vector: List[float], sparse_vector: Dict[str, Any], top_k: int = 5):
        """Query the Pinecone index."""
        return self.index.query(
            namespace=namespace,
            top_k=top_k,
            vector=query_vector,
            sparse_vector=sparse_vector,
            include_metadata=True
        ) 