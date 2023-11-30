from dataclasses import dataclass, field
from typing import Generator, List, Set, Dict, Optional, Tuple, Union, Any
from os.path import sep as PathSep

import llama_index
from llama_index import (
    SimpleDirectoryReader,
    Document,
    Prompt,
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    set_global_service_context,
)
from llama_index.schema import TextNode, NodeWithScore
from llama_index.embeddings import HuggingFaceEmbedding

from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
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
chatbot_instruction = "Solve the problems given below to the best of your ability. Remember, for each wrong answer marks are deducted, hence answer carefully and leave the answer blank and caveat when you are not sure of your solution. \nQuestion: {query_str}"
chatbot_prompt = Prompt(chatbot_instruction)



def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == 'system':
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == 'assistant':
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"
    return prompt

# Loading the model
def load_llm(model_path=MODEL_PATH, colab=False):
    # Uncomment the block below for using with local llm

    llm = LlamaCPP(
        model_path=model_path,
        max_new_tokens=3900,
        temperature=0.5,
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
def get_subject_from_query(agent, query, subjects=subjects):
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
def search_for_para(para: str, top_k: int):
    answers = pipeline.search_one_giant_index(para, top_k=top_k, metadata_key="window")
    return answers


# Personalized helper functions
def create_subjectwise_indexes():
    indexes = {}
    for subject in subjects.keys():
        # Placeholder vector store to be replaced by milvus lite
        indexes[subject] = VectorStoreIndex.from_documents(
            SimpleDirectoryReader(
                input_dir=get_subject_data_path(subject), filename_as_id=True
            ).load_data(),
            storage_context=StorageContext.from_defaults(
                persist_dir=r"vectorstores/{0}".format(subject)
            ),
            show_progress=True,
        )

    return indexes


def create_subjectwise_tools(indexes):
    tools = {}
    for subject in indexes:
        tools[subject] = subject_vector_tool(
            indexes[subject].as_query_engine(), subject
        )
    return tools


def create_chat_agent(llm=load_llm(MODEL_PATH), tools=[], from_dict=False):
    tools = list(tools.values) if from_dict else tools
    return response_agent(llm=llm, tools=tools)


def chat_with_agent(agent: ReActAgent, query):
    chat_response = agent.chat(chatbot_prompt.format(query_str=query))
    return str(chat_response)



llm = load_llm(model_path=MODEL_PATH)
embeddings = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

g_service_ctx = ServiceContext.from_defaults(
    llm=llm, embed_model=embeddings, chunk_size=512
)

pipeline = ingest.AugmentedIngestPipeline(data_dir_path=DATA_PATH, service_context=g_service_ctx)
pipeline.run_pipeline()

if __name__ == "__main__":
    pass
