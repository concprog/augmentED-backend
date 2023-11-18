from dataclasses import dataclass, field
from typing import Generator, List, Set, Dict, Optional, Tuple, Union, Any
from os.path import sep as PathSep

from llama_index import Prompt, ServiceContext, set_global_service_context
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.agent import ReActAgent


import index

from common import *

# TODO Do prompt engineering to fix the instruction and other stuff
chatbot_instruction = "Solve the problems given below to the best of your ability. Remember, for each wrong answer marks are deducted, hence answer carefully and leave the answer blank and caveat when you are not sure of your solution.\nUse the following notes to anwer the question: {context_str}"
chatbot_prompt = Prompt(
    instruct_prompt_template.format(
        instruction=chatbot_instruction, input="{query_str}"
    )
)

# Loading the model
def load_llm(model_path):
    # Load the locally downloaded model here
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

def subject_vector_tool(query_engine, subject):
    vector_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    description=f"Useful for retrieving specific context related to the {subject}",
    )
    return vector_tool


def generate_response_with_rag(vsi,query: str) -> str:
    qa_engine = vsi.as_chat_engine(chat_mode="react")
    ans = qa_engine.chat(
        chatbot_prompt.format(context_str="{context_str}", query_str=query)
    )
    return str(ans)

def response_agent(llm, tools, debug=False):
    agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=debug)
    return agent




if __name__ == "__main__":
    llm = load_llm("chatbot/models/zephyr-7b-beta.Q4_K_M.gguf")
    embeddings = HuggingFaceEmbedding(EMBEDDING_MODEL)

    g_service_ctx = ServiceContext.from_defaults(llm=llm, embed_model=embeddings)
    set_global_service_context(g_service_ctx)

    vsi = index.PersistentDocStoreFaiss(service_context=g_service_ctx).load(DB_FAISS_PATH)
    print(vsi.as_query_engine().query("How should I give therapy to a depressed friend with an anxiety disorder?"))
   
