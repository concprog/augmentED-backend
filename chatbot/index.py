from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import os

DATA_PATH = "data/"
DB_FAISS_PATH = "chatbot/vectorstore/db_faiss"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
EMBEDDING_MODEL_ARGS = model_kwargs = {"device": "cuda"}


# Create vector database
def create_vector_index(overwrite=True):
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=DB_FAISS_PATH)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    if overwrite:
        index.storage_context.persist()
    return index

def load_persistent_index():
    vector_store = FaissVectorStore.from_persist_dir(DB_FAISS_PATH)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=DB_FAISS_PATH)
    index = load_index_from_storage(storage_context=storage_context)
    return index

if __name__ == "__main__":
    create_vector_db()
