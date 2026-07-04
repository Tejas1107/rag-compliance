import os
import shutil
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import fitz  # pymupdf

load_dotenv()

# Clear old ChromaDB
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("Cleared old ChromaDB")

# Extract text page by page using pymupdf
print("Extracting text from PDF...")
doc = fitz.open("fda_q10.pdf")
pages_text = []
for page_num, page in enumerate(doc):
    text = page.get_text()
    if text.strip():  # skip empty pages
        pages_text.append({
            "text": text,
            "page": page_num + 1
        })

print(f"Extracted {len(pages_text)} pages with content")
print(f"\nFirst page preview:")
print(pages_text[0]["text"][:400])

# Convert to TextNodes with page metadata
raw_nodes = [
    TextNode(
        text=p["text"],
        metadata={"page_number": p["page"], "source": "fda_q10.pdf"}
    )
    for p in pages_text
]

# Chunk with sentence-aware splitter
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(raw_nodes)

# Preserve page metadata through chunking
for node in nodes:
    if "page_number" not in node.metadata:
        node.metadata["page_number"] = "unknown"

print(f"\nTotal chunks after splitting: {len(nodes)}")
print(f"\nSample chunk:")
print(nodes[5].text[:300])
print(f"Metadata: {nodes[5].metadata}")

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("fda_compliance")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build index
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True
)

print("\nIngestion complete.")