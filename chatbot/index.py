from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    set_global_service_context,
)
from llama_index import SimpleDirectoryReader
from llama_index.text_splitter import SentenceSplitter
from llama_index.schema import TextNode
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import os

from common import *


class PersistentDocStoreFaiss:
    def __init__(
        self,
        embedding_model_url=EMBEDDING_MODEL,
        embedding_dim=384,
        storage_path=DB_FAISS_PATH,
        data_path: str = DATA_PATH,
        service_context=ServiceContext.from_defaults(llm=None),
    ) -> None:
        self.storage_path = storage_path
        self.data_path = data_path

        self.embedding = HuggingFaceEmbedding(model_name=embedding_model_url)
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        self.storage_ctx = StorageContext.from_defaults(
            vector_store=self.vector_store,
        )
        self.service_ctx = service_context

    def load(self, path):
        self.vector_store = FaissVectorStore.from_persist_dir(path)
        self.storage_ctx = StorageContext.from_defaults(
            vector_store=self.vector_store, persist_dir=path
        )
        self.index = load_index_from_storage(storage_context=self.storage_ctx, service_context=self.service_ctx)
        return self.index

    def create_from_documents(self, documents, save=True):
        storage_ctx = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_ctx,
            service_context=ServiceContext.from_defaults(
                llm=None, embed_model=self.embedding
            ),
        )
        if save:
            self.index.storage_context.persist(persist_dir=self.storage_path)
        return self.index

    def load_or_create(self):
        if os.path.exists(self.storage_path):
            return self.load(self.storage_path)
        else:
            return self.create_from_documents(
                documents=SimpleDirectoryReader(self.data_path).load_data()
            )


if __name__ == "__main__":
    print(os.getcwd())
    documents = SimpleDirectoryReader(get_subject_data_path("psychology")).load_data()
    store = PersistentDocStoreFaiss().create_from_documents(documents=documents)
