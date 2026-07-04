import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
import chromadb
import fitz

load_dotenv()

app = FastAPI(title="FDA Q10 Compliance Query API", description="API for querying FDA Q10 compliance information", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# ---Global query engine---------
query_engine = None

def build_query_engine():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("fda_compliance")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    return index.as_query_engine(similarity_top_k=3, response_model="compact")

@app.on_event("startup")
async def startup_event():
    global query_engine
    query_engine = build_query_engine()
    print("Query engine ready")

# -----Request/Response Models------

class QueryRequest(BaseModel):
    question: str

class SourceChunk(BaseModel):
    score: float
    page: str
    preview: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]

class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int

# -----Endpoints--------

@app.get("/")
def root():
    return {"name": "FDA Compliance RAG API", "status": "running", "endpoints": ["/ingest", "/query", "/health"]}
    
@app.get("/health")
def health():
    return {"status": "healthy", "index_loaded": query_engine is not None}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    global query_engine
    if not query_engine:
        raise HTTPException(status_code=503, detail="Query engine not ready")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    response = query_engine.query(request.question)

    sources = []
    for node in response.source_nodes:
        sources.append(SourceChunk(
            score=round(node.score, 3),
            page=str(node.metadata.get("page_number", "N/A")),
            preview=node.text[:150].replace("\n", " ").strip()
        ))

    return QueryResponse(
        question=request.question,
        answer=str(response),
        sources=sources)

@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    global query_engine
    if not os.path.exists("fda_q10.pdf"):
        raise HTTPException(status_code=400, detail="fda_q10.pdf not found in the current directory")
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        print("Cleared old ChromaDB")
    doc = fitz.open("fda_q10.pdf")
    pages_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages_text.append({
                "text": text,
                "page": page_num + 1
            })
    raw_nodes = [
        TextNode(
            text=p["text"],
            metadata={"page_number": p["page"], "source": "fda_q10.pdf"}
        )
        for p in pages_text
    ]
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(raw_nodes)

    chroma_client = chromadb.PersistentClient(path='./chroma_db')
    chroma_collection = chroma_client.get_or_create_collection("fda_compliance")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage_context)

    query_engine = build_query_engine()
    return IngestResponse(message="Ingestion and indexing completed successfully", chunks_indexed=len(nodes))