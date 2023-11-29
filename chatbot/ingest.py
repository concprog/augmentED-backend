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
from llama_index.embeddings.base import similarity
from llama_index.schema import TextNode, MetadataMode
from llama_index.retrievers import BaseRetriever
from llama_index.query_engine import (
    ToolRetrieverRouterQueryEngine,
    RetrieverQueryEngine,
)
from llama_index.vector_stores import MilvusVectorStore
from llama_index.readers import SimpleDirectoryReader
from llama_index.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.postprocessor import (
    SimilarityPostprocessor,
    MetadataReplacementPostProcessor,
)
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    EntityExtractor,
    BaseExtractor,
)
from milvus import default_server

import os
from typing import List, Dict, Any, Optional


from common import DATA_PATH, EMBEDDING_DIM, EMBEDDING_MODEL, subjects, PathSep, debug


class AugmentedIngestPipeline:
    def __init__(self, data_dir_path: str, service_context: ServiceContext) -> None:
        self.data_dir = data_dir_path
        self.service_ctx = service_context
        self.embed_model = self.service_ctx.embed_model
        self.vector_indexes = {}
        self.metadata_fn = lambda x: {"title": x.replace("_", " ")}
        self.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="og_text",
            include_metadata=True,
        )

    def _load_data(self, path):
        docs = SimpleDirectoryReader(path, file_metadata=self.metadata_fn).load_data()

        return docs

    def _make_nodes(self, docs):
        nodes = self.node_parser.get_nodes_from_documents(docs, show_progress=debug)
        for node in nodes:
            node_embedding = self.embed_model.get_text_embedding(
                node.get_content(metadata_mode=MetadataMode.ALL)
            )
            node.embedding = node_embedding

        return nodes

    def _insert_into_vectorstore(self, subject, nodes, create=True):
        self.collection_name = f"augmentED_{subject}"
        self.vector_store = MilvusVectorStore(
            dim=EMBEDDING_DIM,
            host=default_server.server_address,
            port=default_server.listen_port,
            collection_name=self.collection_name,
            overwrite=create,
        )
        storage_ctx = StorageContext.from_defaults(vector_store=self.vector_store)
        self.vector_indexes[subject] = VectorStoreIndex(
            nodes=nodes,
            service_context=self.service_ctx,
            storage_context=storage_ctx,
        )

    def run_pipeline(self):
        self.one_giant_index_docs = []
        for subject in subjects:
            path = self.data_dir + PathSep + subjects[subject]
            docs = self._load_data(path)
            nodes = self._make_nodes(docs)
            self._insert_into_vectorstore(subject, nodes)
            self.one_giant_index_docs.extend(docs)

        self.one_giant_index = VectorStoreIndex.from_documents(
            self.one_giant_index_docs,
            storage_context=StorageContext.from_defaults(
                persist_dir=r"vectorstores/ogi"
            ),
            show_progress=True,
        )

    def get_indices_as_tools(self):
        tools = []
        for subject in self.vector_indexes:
            vector_tool = QueryEngineTool.from_defaults(
                query_engine=self.vector_indexes[subject].as_query_engine(
                    similarity_top_k=2,
                    node_postprocessors=[
                        MetadataReplacementPostProcessor(target_metadata_key="window")
                    ],
                ),
                description=f"Useful for retrieving specific context for solving questions related to the {subject}",
            )
            tools.append(vector_tool)
        return tools

    def get_subjects_as_query_engines(self) -> Dict:
        self.query_engines = {}
        for subject in subjects:
            self.query_engines[subject] = self.vector_indexes[subject].as_query_engine(
                similarity_top_k=3,
                node_postprocessors=[
                    MetadataReplacementPostProcessor(target_metadata_key="window")
                ],
            )
        return self.query_engines

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
            return list(map(lambda x: x.get_content(metadata_mode=MetadataMode.LLM), answers))

    def ingest_one_file(self, file_path: str, service_context: ServiceContext):
        doc = (
            SimpleDirectoryReader(
                input_files=[file_path], filename_as_id=True
            ).load_data(),
        )
        nodes = self.node_parser.get_nodes_from_documents(doc, show_progress=debug)
        for node in nodes:
            node_embedding = self.service_ctx.embed_model.get_text_embedding(
                node.get_content(metadata_mode=MetadataMode.ALL)
            )
            node.embedding = node_embedding

        index = VectorStoreIndex(
            nodes = nodes,
            storage_context=StorageContext.from_defaults(
                persist_dir=r"vectorstores/{0}".format(file_path.partition(".")[0])
            ),
            show_progress=True,
        )
        return index


if __name__ == "__main__":
    default_server.start()
    print(os.getcwd())
    pipe = AugmentedIngestPipeline(
        DATA_PATH,
        service_context=ServiceContext.from_defaults(
            llm=None, embed_model=HuggingFaceEmbedding(EMBEDDING_MODEL)
        ),
    )
    pipe.run_pipeline()
    pipe.search_one_giant_index("depr")
    default_server.stop()
