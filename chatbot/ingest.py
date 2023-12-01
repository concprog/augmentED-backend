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
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.schema import TextNode, MetadataMode
from llama_index.vector_stores import MilvusVectorStore
from llama_index.readers import SimpleDirectoryReader
from llama_index.node_parser import SentenceWindowNodeParser, SentenceSplitter, SimpleNodeParser
from llama_index.postprocessor import (
    SimilarityPostprocessor,
    MetadataReplacementPostProcessor,
)

from milvus import default_server

import os
from typing import List, Dict, Any, Optional


from chatbot.common import DATA_PATH, EMBEDDING_DIM, EMBEDDING_MODEL, SIMILARITY_SEARCH_THRESHOLD, path_leaf, subjects, PathSep, debug


class AugmentedIngestPipeline:
    def __init__(
        self, data_dir_path: str, service_context: ServiceContext,
    ) -> None:
        self.data_dir = data_dir_path
        self.service_ctx = service_context
        self.embed_model = self.service_ctx.embed_model
        self.vector_indexes = {}
        self.metadata_fn = lambda x: {"title": x.replace("_", " ")}
        self.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
            include_metadata=True,
        )
        self.create = False

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
            overwrite=False
        )

        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

        self.vector_indexes[subject] = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            service_context=self.service_ctx,
            storage_context=storage_ctx,
        )

    def _get_subject_query_engine(self, subject):
        query_engine = self.vector_indexes[subject].as_query_engine(
            similarity_top_k=3,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=SIMILARITY_SEARCH_THRESHOLD),
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )
        return query_engine

    def _get_subject_chat_engine(self, subject):
        query_engine = self.vector_indexes[subject].as_chat_engine(
            mode="context",
            similarity_top_k=2,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=SIMILARITY_SEARCH_THRESHOLD),
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )
        return query_engine


    def run_pipeline(self, create=False):
        self.create = create
        if self.create:
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
                subject="OneGiantIndex", nodes=self.one_giant_index_nodes, create=self.create
            )

        else:
            for subject in subjects:
                self._load_vectorstore(subject)
            self._load_vectorstore("OneGiantIndex")

        self.one_giant_index = self.vector_indexes["OneGiantIndex"]
        self.query_everything = self._get_subject_query_engine("OneGiantIndex")
    
    def search_one_giant_index(
        self,
        query,
        top_k=10,
        replace_with_meta=True,
        metadata_key="title",
    ):
        retr = self.one_giant_index.as_retriever(
            similarity_top_k=top_k,
        )
        answers = retr.retrieve(query)
        if replace_with_meta:
            return list(set(map(lambda x: x.metadata[metadata_key], answers)))
        else:
            return list(
                map(lambda x: x.get_content(metadata_mode=MetadataMode.LLM), answers)
            )

    def query_one_file(self,file_path):
        docs = SimpleDirectoryReader(
            input_files=[file_path], file_metadata=self.metadata_fn, filename_as_id=True
        ).load_data()
        nodes = self._make_nodes(docs)
        self._insert_into_vectorstore("UserUploadedDocument", nodes)
        self._insert_into_vectorstore("OneGiantIndex", nodes)
        return self._get_subject_query_engine("UserUploadedDocument")
    

class SimpleIngestPipeline:
    def __init__(
        self, data_dir_path: str, service_context: ServiceContext, create=False
    ) -> None:
        self.data_dir = data_dir_path
        self.service_ctx = service_context
        self.embed_model = self.service_ctx.embed_model
        self.vector_indexes = {}
        self.metadata_fn = lambda x: {"title": path_leaf(x)}
        self.node_parser = SimpleNodeParser(chunk_size=512)
        self.create = create

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
            overwrite=False
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

    def run_pipeline(self):
        if self.create:
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
                subject="OneGiantIndex", nodes=self.one_giant_index_nodes, create=self.create
            )

        else:
            for subject in subjects:
                self._load_vectorstore(subject)
            self._load_vectorstore("OneGiantIndex")

        self.one_giant_index = self.vector_indexes["OneGiantIndex"]


if __name__ == "__main__":
    pipe = AugmentedIngestPipeline(
        data_dir_path=DATA_PATH,
        service_context=ServiceContext.from_defaults(
            llm=None, embed_model=HuggingFaceEmbedding(EMBEDDING_MODEL)
        ),
    )
    pipe.run_pipeline(create=True)
