from dataclasses import dataclass, field
from typing import Generator, List, Set, Dict, Optional, Tuple, Union, Any
from os.path import sep as PathSep

from transformers import AutoTokenizer

import llama_index
from llama_index import (
    PromptTemplate,
    Document,
    Prompt,
    ServiceContext,
    set_global_service_context,
    set_global_tokenizer
)

from llama_index.response_synthesizers import TreeSummarize
from llama_index.retrievers import BM25Retriever
from llama_index.schema import TextNode, NodeWithScore
from llama_index.embeddings import HuggingFaceEmbedding

from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    completion_to_prompt,
)
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.agent import ReActAgent



from chatbot import ingest

from chatbot.common import *

# TODO Do prompt engineering to fix the instruction and other stuff

###########
# Prompts #
###########
chatbot_instruction = "Solve the problems given below to the best of your ability. Remember, for each wrong answer, you will be penalized - hence answer carefully and leave the answer blank or caveat when you are not sure of your solution. \nQuestion: {query_str}"
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


    set_global_tokenizer(
        AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").encode
    )

    llm = LlamaCPP(
        model_path=model_path,
        context_window=5120,
        max_new_tokens=1536,
        temperature=0.5,
        model_kwargs={"n_gpu_layers": 24 if not colab else 64},
        messages_to_prompt=messages_to_prompt,
        verbose=True,
    )
    return llm




# LLM task helpers

def build_input_prompt(message, system_prompt):
    """
    Constructs the input prompt string from the chatbot interactions and the current message.
    """
    input_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n"
    input_prompt = input_prompt + str(message) + "</s>\n<|assistant|>"
    return input_prompt

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

def search_doc_metadata(docs: List[Document], query: str, metadata_key: str, top_k=10,keep_duplicates=False):
    meta_nodes = list(map(lambda x: TextNode(text=x.metadata[metadata_key]), docs))
    if not keep_duplicates:
        meta_nodes = list(set(meta_nodes))
    retr = BM25Retriever.from_defaults(nodes=meta_nodes,similarity_top_k=top_k)
    answers = retr.retrieve(query)
    return list(set(map(lambda x: x.get_content(metadata_mode="all"), answers)))

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

# Personalized helper functions

def create_tools(indexes):
    tools = []
    for subject in indexes:
        tools.append(subject_vector_tool(indexes[subject], subject))
    return tools

def create_chat_agent(llm=load_llm(MODEL_PATH), tools=[], from_dict=False):
    tools = list(tools.values) if from_dict else tools
    return response_agent(llm=llm, tools=tools)


def chat_with_agent(agent: ReActAgent, query):
    chat_response = agent.chat(chatbot_prompt.format(query_str=query))
    return str(chat_response)

def summarize_text(text, paras=["<no context present>"]):
    
    custom_prompt_tmpl = (
        "<|system|>\n"
        "Summarize the provided book or paragraph, emphasizing key concepts and minimizing unnecessary details. Be concise and provide the essence of the content in the least space possible in points.</s>\n"
        "<|user|>\n"
        "Do not summarize the following context, instead use them to decide what topics are important and which ones are unnecessary: "
        "{context_str}"
        "Summarize the following paragraphs only, concisely: "
        "{query_str} </s>"
        "<|assistant|>"
    )
    custom_prompt = PromptTemplate(custom_prompt_tmpl)    
    summarizer = TreeSummarize(verbose=True, summary_template=custom_prompt)
    response = summarizer.get_response(f"{text}",paras)  # Empty query
    return (str(response))


llm = load_llm(model_path=MODEL_PATH)
embeddings = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

g_service_ctx = ServiceContext.from_defaults(
    llm=llm, embed_model=embeddings,
)

everything_pipeline = ingest.AugmentedIngestPipeline(data_dir_path=DATA_PATH, service_context=g_service_ctx)
everything_pipeline.run_pipeline()



# pipeline fn.s 

def search_for_title(title: str, top_k: int) -> List[str]:
    results = everything_pipeline.search_one_giant_index(title, top_k=top_k, metadata_key="title")
    return results

def search_for_paras(para: str, top_k: int):
    answers = everything_pipeline.search_one_giant_index(para, top_k=top_k, metadata_key="window")
    return answers

def augmented_summarize(text: str, top_k:int = 2):
    paras = search_for_paras(text, top_k)
    summary = summarize_text(text, paras)
    return summary

##
## externally accessible fn.s and variables
##

tools = create_tools(everything_pipeline.vector_indexes)
agent = create_chat_agent(llm=llm, tools=tools)

if __name__ == "__main__":
    set_global_service_context(g_service_ctx)
    print(augmented_summarize("Who is rogers?"))
    # <|user|>How do I fix my friend's crippling anxiety and depression?\nYou know that {context_str}</s><|assistant|>