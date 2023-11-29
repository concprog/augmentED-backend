from llama_index import VectorStoreIndex, set_global_service_context

import model
from common import *

from pydantic import BaseModel

set_global_service_context(model.g_service_ctx)


document_chat_engine_cache = []

indexes = model.create_subjectwise_indexes()
tools = model.create_subjectwise_tools(indexes)
agent = model.create_chat_agent(llm=model.load_llm(MODEL_PATH), tools=tools, from_dict=True)

# Accessible fn.s

def search_passages(passage, top_k=3):
    subject_index = indexes[model.get_subject_from_query(agent, passage, subjects)]
    similar_chunks = model.vector_search(passage, subject_index, top_k)
    chunk_with_docid = [(chunk.get_content(), chunk.node_id)for chunk in similar_chunks]


# Responses
def generate_generic_response(query):
    response = agent.chat(model.chatbot_prompt.format(query_str=query))
    return str(response)

def set_document_chat_engine(file_path):
    doc_chat_engine = model.pipeline.index_one_doc(file_path).as_chat_engine()
    if len(document_chat_engine_cache) > 2:
        document_chat_engine_cache.pop()
    document_chat_engine_cache.append(doc_chat_engine)

def chat_with_document(query):
    doc_chat_engine = chat_with_document[0]
    return doc_chat_engine.



def generate_openai_from_response(response):
    choices = []

    choices.append(
        {
            "role": "assistant",
            "content": response,
        }
    )
    data = {"choices": choices}
    return data


# Search


def search_catalogue(query, top_k=10):
    pass


if __name__ == "__main__":
    generate_generic_response(
        "How do I help my friend who is suffering from severe anxiety and depression?"
    )
