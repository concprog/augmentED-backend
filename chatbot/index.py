from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index import SimpleDirectoryReader
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import os

from llama_index.vector_stores.types import DEFAULT_PERSIST_FNAME

from common import *


class PersistentDocStoreFaiss:
    def __init__(
        self,
        embedding_model=HuggingFaceEmbedding(model_name=EMBEDDING_MODEL),
        embedding_dim=EMBEDDING_DIM,
        storage_path=DB_FAISS_PATH,
        data_path: str = DATA_PATH,
        service_context=None,
    ) -> None:
        self.storage_path = storage_path
        self.data_path = data_path

        self.embedding = embedding_model
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        self.storage_ctx = StorageContext.from_defaults(
            vector_store=self.vector_store,
        )
        self.service_ctx = (
            service_context
            if service_context is not None
            else ServiceContext.from_defaults(llm=None, embed_model=self.embedding)
        )

    def load_from_storage(self):
        self.vector_store = FaissVectorStore.from_persist_dir(persist_dir=self.storage_path)
        storage_ctx = StorageContext.from_defaults(
            vector_store=self.vector_store, persist_dir=self.storage_path
        )
        self.index = load_index_from_storage(
            storage_context=storage_ctx, service_context=self.service_ctx, show_progress=True
        )
        return self.index

    def create_from_documents(self, documents, save=True):
        storage_ctx = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_ctx,
            service_context=self.service_ctx,

            show_progress=True
        )
        if save:
            self.index.storage_context.persist(persist_dir=self.storage_path)
            # self.vector_store.persist(persist_path=os.path.join(self.storage_path, DEFAULT_PERSIST_FNAME))
        return self.index

    def load_or_create(self):
        if os.path.isdir(self.storage_path) and os.listdir(self.storage_path) != []:
            return self.load_from_storage()
        else:
            return self.create_from_documents(
                documents=SimpleDirectoryReader(
                    self.data_path, filename_as_id=True
                ).load_data()
            )


if __name__ == "__main__":
    print(os.getcwd())
    docs = SimpleDirectoryReader(get_subject_data_path("psychology"), filename_as_id=True).load_data()
    store = PersistentDocStoreFaiss(data_path=get_subject_data_path("psychology")).load_or_create()
    
