import os
import logging
import yaml
import nltk
from typing import List, Dict, Any

from .config.settings import PINECONE_INDEX_NAME
from .models.embeddings import get_dense_embeddings, load_embedding_model
from .models.bm25 import BM25Singleton, save_bm25_instance, load_bm25_instance
from .services.pdf_service import PDFExtractor, get_text_chunks
from .services.pinecone_service import PineconeService
from .services.llm_service import LLMService

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Download NLTK data
nltk.download("punkt_tab")

def load_prompts(file_path: str = "components/prompts.yaml") -> Dict[str, str]:
    """Load prompts from YAML file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Process PDF file and extract text chunks."""
    filename = os.path.splitext(os.path.basename(file_path))[0].lower()
    
    # Extract PDF data
    pdf_extractor = PDFExtractor(file_path)
    pdf_data = pdf_extractor.get_extracted_data()
    
    if pdf_data:
        return get_text_chunks(filename, pdf_data)
    return []

def initialize_services(index_name: str):
    """Initialize required services."""
    pinecone_service = PineconeService(index_name)
    llm_service = LLMService()
    embedding_model = load_embedding_model()
    return pinecone_service, llm_service, embedding_model

def rag_pipeline(query: str, context: str, llm_service: LLMService, prompts: Dict[str, str]) -> str:
    """Run the RAG pipeline to generate an answer."""
    enhanced_query = llm_service.enhance_query(query, prompts["QUERY_REWRITER"])
    answer = llm_service.generate_answer(enhanced_query, context, prompts["RAG_GENERATE_ANSWER"])
    return answer

def chatbot(pinecone_service: PineconeService, llm_service: LLMService, 
           embedding_model: Any, bm25_instance: Any, namespace: str, prompts: Dict[str, str]):
    """Run the chatbot interface."""
    print("Welcome to the RAG Chatbot! Type 'exit' to quit.")
    
    while True:
        user_query = input("\nYour question: ")
        
        if user_query.lower() in ["exit", "quit"]:
            print("\nThanks for chatting, hope this was helpful!")
            break
            
        # Get dense and sparse embeddings
        dense_query = get_dense_embeddings([user_query])[0]
        sparse_query = bm25_instance.encode(user_query)
        
        # Query Pinecone
        results = pinecone_service.query(
            namespace=namespace,
            query_vector=dense_query,
            sparse_vector=sparse_query
        )
        
        # Extract context from results
        context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
        
        # Generate answer
        answer = rag_pipeline(user_query, context, llm_service, prompts)
        
        print("\n" + "="*100)
        print("ANSWER:\n")
        print(answer)
        print("="*100)

def main():
    # Load configuration
    prompts = load_prompts()
    
    # Initialize services
    pinecone_service, llm_service, embedding_model = initialize_services(PINECONE_INDEX_NAME)
    
    # Process command line arguments
    import sys
    to_upsert = len(sys.argv) > 1 and sys.argv[1].lower() in ["yes", "y"]
    
    file_path = "/Documents/Tutorial04_Solution.pdf"
    filename = os.path.splitext(os.path.basename(file_path))[0].lower()
    namespace = filename
    pickle_path = f"components/hybrid-rag/bm25_{filename}.pkl"
    
    if to_upsert:
        # Process PDF and create embeddings
        text_chunks = process_pdf(file_path)
        node_texts = [chunk["Text"].lower() for chunk in text_chunks]
        
        # Initialize and save BM25
        bm25_instance = BM25Singleton()
        bm25_instance.fit(node_texts)
        save_bm25_instance(bm25_instance, pickle_path)
        
        # Create vectors and upsert to Pinecone
        vectors = []
        for i, text in enumerate(node_texts):
            dense_embedding = get_dense_embeddings([text])[0]
            sparse_embedding = bm25_instance.encode(text)
            
            vectors.append({
                "id": f"vector{i+1}",
                "values": dense_embedding,
                "sparse_values": sparse_embedding,
                "metadata": {"text": text}
            })
        
        success = pinecone_service.upsert_vectors(vectors, namespace)
        if not success:
            logging.error("Failed to upsert vectors to Pinecone")
            return
    
    # Load existing BM25 model
    bm25_instance = load_bm25_instance(pickle_path)
    
    # Start chatbot
    chatbot(pinecone_service, llm_service, embedding_model, bm25_instance, namespace, prompts)

if __name__ == "__main__":
    main()