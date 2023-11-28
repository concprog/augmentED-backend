from re import sub
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    service_context,
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.schema import TextNode
from llama_index.retrievers import BaseRetriever
from llama_index.query_engine import ToolRetrieverRouterQueryEngine, RetrieverQueryEngine
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


from common import *

default_server.start()


class AugmentedIngestPipeline:
    def __init__(self, data_dir_path: str, service_context: ServiceContext) -> None:
        self.data_dir = data_dir_path
        self.service_ctx = service_context
        self.embed_model = self.service_ctx.embed_model
        self.vector_indexes = {}
        self.metadata_fn = lambda x: {"title":x.replace("_", " ")}

    def _load_data(self, path):


        docs = SimpleDirectoryReader(
            path, 
            file_metadata=self.metadata_fn
        ).load_data()

        return docs


    def _make_nodes(self, docs):

        text_parser = SentenceSplitter(chunk_size=512)

        text_chunks = []
        # maintain relationship with source doc index, to help inject doc metadata in (3)
        doc_idxs = []

        for doc_idx, doc in enumerate(docs):
            cur_text_chunks = text_parser.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))

        
        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(
                text=text_chunk,
            )
            src_doc = docs[doc_idxs[idx]]
            node.metadata = src_doc.metadata
            nodes.append(node)

        for node in nodes:
            node_embedding = self.embed_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding
        
        return nodes

    def _insert_into_vectorstore(self, subject, nodes,create=True):
        self.collection_name = 'augmentED'
        self.vector_store = MilvusVectorStore(
            uri=r"{host}:{port}".format(
                host=default_server.server_address, port=default_server.listen_port
            ),
            collection_name=self.collection_name,
            overwrite=create,
        )
        storage_ctx = StorageContext.from_defaults(vector_store=self.vector_store)
        self.vector_indexes[subject]=VectorStoreIndex(nodes=nodes, service_context=self.service_ctx, storage_context=storage_ctx)
    
    def run_pipeline(self):
        for subject in subjects:
            path = self.data_dir+PathSep+subjects[subject]
            docs = self._load_data(path)
            nodes = self._make_nodes(docs)
            self._insert_into_vectorstore(subject, nodes)

    def get_indices_as_tools(self):
        tools = []
        for subject in self.vector_indexes:
            vector_tool = QueryEngineTool.from_defaults(
                query_engine=self.vector_indexes[subject],
                description=f"Useful for retrieving specific context for anything related to the {subject}",
            )
            tools.append(vector_tool)
        return tools
            

        


if __name__ == "__main__":
    print(os.getcwd())
    docs = SimpleDirectoryReader(
        get_subject_data_path("psychology"), filename_as_id=True
    ).load_data()
