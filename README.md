# FDA Compliance RAG API

A retrieval-augmented generation (RAG) pipeline over the FDA Q10 
Pharmaceutical Quality System guidance document.

## Architecture

PDF Document → PyMuPDF text extraction → SentenceSplitter chunking (512 tokens, 50 overlap)

→ OpenAI text-embedding-ada-002 → ChromaDB vector store

→ FastAPI query endpoint → top-3 chunk retrieval → GPT-3.5-turbo response

→ JSON response with answer + source citations + page numbers

## Stack
- **LlamaIndex** — RAG orchestration
- **ChromaDB** — persistent vector store
- **PyMuPDF (fitz)** — PDF text extraction
- **OpenAI** — embeddings (ada-002) + LLM (gpt-3.5-turbo)
- **FastAPI** — REST API layer
- **Uvicorn** — ASGI server

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Index status |
| POST | `/query` | Ask a question against the document |
| POST | `/ingest` | Re-ingest the PDF |

## Running locally

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
# Add OPENAI_API_KEY to .env
python ingest.py       # Run once to build vector store
uvicorn main:app --reload
# Visit http://127.0.0.1:8000/docs
```

## Example query

```json
POST /query
{
  "question": "What are the key elements of a pharmaceutical quality system?"
}
```

```json
{
  "question": "What are the key elements of a pharmaceutical quality system?",
  "answer": "The key elements include quality policy, management responsibilities, 
             process performance monitoring, CAPA, change management...",
  "sources": [
    { "score": 0.81, "page": "8", "preview": "..." }
  ]
}
```
