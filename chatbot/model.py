from dataclasses import dataclass, field
from typing import Generator, List, Set, Dict, Optional, Tuple, Union, Any
from os.path import sep as PathSep

import llama_index
from llama_index import (
    Document,
    Prompt,
    ServiceContext,
    VectorStoreIndex,
    set_global_service_context,
)
from llama_index.schema import Node, NodeWithScore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.retrievers import BM25Retriever
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.agent import ReActAgent


import index

from common import *
from common import EMBEDDING_MODEL

# TODO Do prompt engineering to fix the instruction and other stuff
chatbot_instruction = "Solve the problems given below to the best of your ability. Remember, for each wrong answer marks are deducted, hence answer carefully and leave the answer blank and caveat when you are not sure of your solution.\nUse the following notes to anwer the question: {context_str}"
chatbot_prompt = Prompt(
    instruct_prompt_template.format(
        instruction=chatbot_instruction, input="{query_str}"
    )
)


# Loading the model
def load_llm(model_path=MODEL_PATH):
    llm = LlamaCPP(
        model_path=model_path,
        max_new_tokens=2048,
        temperature=0.7,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 18},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    return llm


# Tools and Agent defn.s


def subject_vector_tool(query_engine, subject):
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        description=f"Useful for retrieving specific context related to the {subject}",
    )
    return vector_tool


def response_agent(llm, tools, debug=False):
    agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=debug)
    return agent


# Search (vector, bm25, ensemble)


def vector_search(query: str, vsi: VectorStoreIndex, n=10) -> List[NodeWithScore]:
    retr = vsi.as_retriever(similarity_top_k=n)
    docs = retr.retrieve(query)
    return docs


def bm25_search(query: str, nodes: List[Node], n=10):
    retr = BM25Retriever.from_defaults()
    return


llm = load_llm(model_path=MODEL_PATH)
embeddings = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

g_service_ctx = ServiceContext.from_defaults(
    llm=llm, embed_model=embeddings, chunk_size=1024
)
set_global_service_context(g_service_ctx)

if __name__ == "__main__":
    vsi = index.PersistentDocStoreFaiss(
        service_context=g_service_ctx, storage_path=DB_FAISS_PATH
    ).load_from_storage()
    qe = vsi.as_query_engine()
    print(
        qe.query(
            "How should I give therapy to a depressed friend with an anxiety disorder? I know that {context_str}"
        )
    )
