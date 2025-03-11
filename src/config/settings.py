import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# API URLs
HF_EMBEDDER_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN =  os.getenv("HF_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
HF_EMBEDDING_SIMILARITY_API = os.getenv("HF_EMBEDDING_SIMILARITY_API")

# Model Settings
CHAT_MODEL = "llama3-70b-8192"
PINECONE_INDEX_NAME = "rag-testing-hybrid"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# File Paths
PDF_DIR = "../PDF"
DATA_DIR = "../data"
COMPONENTS_DIR = "components"

# Create directories if they don't exist
for dir_path in [PDF_DIR, DATA_DIR, COMPONENTS_DIR]:
    os.makedirs(dir_path, exist_ok=True) 