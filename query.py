import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()

#reconnect to existing chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("fda_compliance")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
query_engine = index.as_query_engine(similarity_top_k=3, response_model="compact")

questions = ["What are the key elements of a pharmaceutical quality system?",
    "What does Q10 say about management responsibility?",
    "How should continual improvement be handled?"]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print(f"A: {response}")
    print(f"\nSources:")
    for node in response.source_nodes:
        print(f" - Score: {node.score: .3f}, | Page: {node.metadata.get('page_label', 'N/A')}")
    print("-" * 60)
