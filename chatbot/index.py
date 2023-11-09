from llama_index import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index import SimpleDirectoryReader
from llama_index.text_splitter import SentenceSplitter
from llama_index.schema import TextNode
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import os

from config import *

DATA_PATH = "chatbot/data/"
DB_FAISS_PATH = "chatbot/vectorstore/db_faiss"


documents = SimpleDirectoryReader(DATA_PATH).load_data()


class PersistentDocStoreFaiss:
    def __init__(
        self,
        embedding_model_url=EMBEDDING_MODEL,
        embedding_dim=384,
        storage_path=DB_FAISS_PATH,
    ) -> None:
        self.storage_path = storage_path
        self.embedding = HuggingFaceEmbedding(model_name=embedding_model_url)
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        self.storage_ctx = StorageContext.from_defaults(
            vector_store=self.vector_store, 
        )
        self.index = None

    def load(self):
        self.vector_store = FaissVectorStore.from_persist_dir(self.storage_path)
        self.storage_ctx = StorageContext.from_defaults(
            vector_store=self.vector_store, persist_dir=self.storage_path
        )
        self.index = load_index_from_storage(storage_context=self.storage_ctx)
        return self.index

    def create(self, documents, data_path=DATA_PATH, save=True):
        text_splitter = SentenceSplitter(
            chunk_size=1024,
            # separator=" ",
        )
        text_chunks = []

        # maintain relationship with source doc index, to help inject doc metadata in (3)
        doc_idxs = []
        for doc_idx, doc in enumerate(documents):
            cur_text_chunks = text_splitter.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))

        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(
                text=text_chunk,
            )
            src_doc = documents[doc_idxs[idx]]
            node.metadata = src_doc.metadata
            nodes.append(node)

        for node in nodes:
            node_embedding = self.embedding.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding

        self.vector_store.add(nodes)
        self.index = VectorStoreIndex(storage_context=self.storage_ctx)
        if save:
            self.index.storage_context.persist(persist_dir=self.storage_path)
        return self.index

    def load_or_create_default(self):
        if os.path.exists(self.storage_path):
            return self.load()
        else:
            return self.create(documents=SimpleDirectoryReader(DATA_PATH).load_data())


if __name__ == "__main__":
    documents = SimpleDirectoryReader(DATA_PATH).load_data()
    store = PersistentDocStoreFaiss().create(documents=documents)
