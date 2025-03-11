# RAG Model with Hybrid Search

This project implements a Retrieval-Augmented Generation (RAG) model with hybrid search capabilities, combining dense and sparse embeddings for improved document retrieval and question answering.

## Features

- PDF text extraction using Adobe PDF Services
- Hybrid search using dense (HuggingFace) and sparse (BM25) embeddings
- Vector storage with Pinecone
- Question answering using Groq LLM
- Query enhancement for better retrieval

## Prerequisites

- Python 3.8+
- Required API keys:
  - Pinecone API key
  - HuggingFace API key
  - Groq API key
  - Adobe PDF Services credentials

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG_Lenor
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Copy `.env.example` to `.env`
- Fill in your API keys and credentials

## Project Structure

```
RAG_Lenor/
├── src/
│   ├── config/
│   │   └── settings.py
│   ├── models/
│   │   ├── embeddings.py
│   │   └── bm25.py
│   ├── services/
│   │   ├── pdf_service.py
│   │   ├── pinecone_service.py
│   │   └── llm_service.py
│   └── main.py
├── requirements.txt
├── .env
└── README.md
```

## Usage

1. Place your PDF file in the `PDF` directory.

2. Run the script with the upsert flag to process and index a new document:
```bash
python src/main.py yes
```

3. Run without the flag to use existing indexed data:
```bash
python src/main.py
```

4. Type your questions in the interactive chat interface.

## API Keys Setup

1. Pinecone API Key:
   - Sign up at https://www.pinecone.io/
   - Create a project and get API key from dashboard

2. HuggingFace API Key:
   - Sign up at https://huggingface.co/
   - Get API key from Settings -> Access Tokens

3. Groq API Key:
   - Sign up at https://console.groq.com/
   - Get API key from dashboard

4. Adobe PDF Services:
   - Sign up at https://developer.adobe.com/document-services/apis/pdf-services/
   - Create credentials and get client ID and secret

## Contributing

Feel free to submit issues and enhancement requests!