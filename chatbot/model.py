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
from llama_index.retrievers import RecursiveRetriever
from llama_index.llms import LlamaCPP, OpenAI
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.agent import ReActAgent


import ingest

from common import *

# TODO Do prompt engineering to fix the instruction and other stuff

###########
# Prompts #
###########
chatbot_instruction = "Solve the problems given below to the best of your ability. Remember, for each wrong answer marks are deducted, hence answer carefully and leave the answer blank and caveat when you are not sure of your solution.\nUse the following notes to anwer the question: {context_str}"
chatbot_prompt = Prompt(
    instruct_prompt_template.format(
        instruction=chatbot_instruction, input="{query_str}"
    )
)


# Loading the model
def load_llm(model_path=MODEL_PATH, colab=False):

    # Uncomment the block below for using with local llm

    llm = LlamaCPP(
        model_path=model_path,
        max_new_tokens=3900,
        temperature=0.7,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 18 if not colab else 64},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    return llm


# Tools and Agent defn.s and helpers


def subject_vector_tool(query_engine, subject):
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        description=f"Useful for retrieving specific context for anything related to the {subject}",
    )
    return vector_tool


def response_agent(llm, tools, debug=False):
    agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=debug)
    return agent


# LLM task helpers
def get_subject_from_query(agent, query, subjects = subjects):
    fmted_subjects = ", ".join(list(subjects.keys()))
    generate_responses = lambda x: str(agent.chat(x))
    subject = generate_responses(
        f"Of the given subjects {fmted_subjects}, which subject does the question '{query}' pertain to? Answer iOf the given subjects {fmted_subjects}, which subject does the question '{query}' pertain to? Answer in a single word containing the name of the subject.n a single word containing the name of the subject."
    )
    if subject not in subjects:
        subject = generate_responses(
            (
                f"Given the query '{query}', you classified it as a {subject} question. However, that is an incorrect answer. "
                f"So, keeping that in mind, classify it into one of the following categories: {fmted_subjects}. Answer in a single word containing the name of the subject."
            )
        )

    return subject



# Search (vector, bm25, ensemble)


def vector_search(query: str, vsi: VectorStoreIndex, n=10) -> List[NodeWithScore]:
    retr = vsi.as_retriever(similarity_top_k=n)
    docs = retr.retrieve(query)
    return docs


llm = load_llm(model_path=MODEL_PATH)
embeddings = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

g_service_ctx = ServiceContext.from_defaults(
    llm=llm, embed_model=embeddings, chunk_size=512
)


if __name__ == "__main__":
    pass