from importlib import metadata
from pathlib import WindowsPath
from re import sub
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    StorageContext, 
    load_index_from_storage,
    global_service_context,
)
import llama_index
from llama_index.ingestion import IngestionPipeline, IngestionCache
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.schema import TextNode, MetadataMode
from llama_index.vector_stores import MilvusVectorStore
from llama_index.readers import SimpleDirectoryReader
from llama_index.node_parser import SentenceWindowNodeParser, SimpleNodeParser
from llama_index.text_splitter import SentenceSplitter
from llama_index.postprocessor import (
    SimilarityPostprocessor,
    MetadataReplacementPostProcessor,
)

from milvus import default_server

import os
from typing import List, Dict, Any, Optional


from chatbot.common import (
    DATA_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    SIMILARITY_SEARCH_THRESHOLD,
    path_leaf,
    subjects,
    PathSep,
    debug,
)



class SimpleIngestPipeline:
    def __init__(
        self, data_dir_path: str, service_context: ServiceContext
    ) -> None:
        self.data_dir = data_dir_path
        self.service_ctx = service_context
        self.embed_model = self.service_ctx.embed_model
        self.vector_indexes = {}
        self.metadata_fn = lambda x: {"title": path_leaf(x)}
        self.node_parser = SimpleNodeParser(chunk_size=512)


    def _load_data(self, path):
        docs = SimpleDirectoryReader(
            path, file_metadata=self.metadata_fn, filename_as_id=True
        ).load_data()
        return docs

    def _make_nodes(self, docs):
        nodes = self.node_parser.get_nodes_from_documents(docs, show_progress=debug)
        return nodes

    def _insert_into_vectorstore(self, subject, nodes, create=False):
        collection_name = f"augmentED_{subject}"
        vector_store = MilvusVectorStore(
            dim=EMBEDDING_DIM,
            host="127.0.0.1",
            port=default_server.listen_port,
            collection_name=collection_name,
            overwrite=create,
        )

        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

        self.vector_indexes[subject] = VectorStoreIndex(
            nodes=nodes,
            service_context=self.service_ctx,
            storage_context=storage_ctx,
        )

    def _load_vectorstore(self, subject):
        collection_name = f"augmentED_{subject}"
        vector_store = MilvusVectorStore(
            dim=EMBEDDING_DIM,
            host="127.0.0.1",
            port=default_server.listen_port,
            collection_name=collection_name,
            overwrite=False,
        )

        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

        self.vector_indexes[subject] = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            service_context=self.service_ctx,
            storage_context=storage_ctx,
        )

    def _get_subject_query_engine(self, subject) -> Dict:
        query_engine = self.vector_indexes[subject].as_query_engine(
            similarity_top_k=3,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )
        return query_engine

    def run_pipeline(self, create: bool = False):
        if create:
            self.one_giant_index_nodes = []
            self.all_docs = []
            for subject in subjects:
                path = self.data_dir + PathSep + subjects[subject]

                docs = self._load_data(path)
                nodes = self._make_nodes(docs)
                self._insert_into_vectorstore(subject=subject, nodes=nodes)

                self.one_giant_index_nodes.extend(nodes)
                self.all_docs.extend(docs)

            self._insert_into_vectorstore(
                subject="OneGiantIndex",
                nodes=self.one_giant_index_nodes,
                create=create,
            )

        else:
            for subject in subjects:
                self._load_vectorstore(subject)
            self._load_vectorstore("OneGiantIndex")

        self.one_giant_index = self.vector_indexes["OneGiantIndex"]

class AugmentEDIngestPipeline:
    def __init__(self, **kwargs) -> None:
        pass


if __name__ == "__main__":
    pipe = SimpleIngestPipeline(
        data_dir_path=DATA_PATH,
        service_context=ServiceContext.from_defaults(
            llm=None, embed_model=HuggingFaceEmbedding(EMBEDDING_MODEL)
        ),
    )
    pipe.run_pipeline(create=True)
