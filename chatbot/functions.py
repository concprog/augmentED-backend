from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, set_global_service_context
from llama_index.tools import QueryEngineTool

import model
from common import *

set_global_service_context(model.g_service_ctx)
def create_subjectwise_indexes():
    indexes = {}
    for subject in subjects.keys():
        # Placeholder vector store to be replaced by milvus lite
        indexes[subject] = VectorStoreIndex.from_documents(
            SimpleDirectoryReader(
                input_dir=get_subject_data_path(subject),
                filename_as_id=True
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
        tools[subject] = model.subject_vector_tool(indexes[subject].as_query_engine(), subject)
    return tools


def create_chat_agent(llm=model.load_llm(MODEL_PATH), tools=[], from_dict=False):
    tools = list(tools.values) if from_dict else tools
    return model.response_agent(llm=llm, tools=tools)


indexes = create_subjectwise_indexes()
tools = create_subjectwise_tools(indexes)
agent = create_chat_agent(llm=model.load_llm(MODEL_PATH), tools=tools, from_dict=True)

# Accessible fn.s

def search_passages(passage, top_k=3):
    subject_index = indexes[model.get_subject_from_query(agent, passage, subjects)]
    similar_chunks = model.vector_search(passage, subject_index, top_k)
    chunk_with_docid = [(chunk.get_content(), chunk.node_id)for chunk in similar_chunks]


# Responses
def generate_responses(query):
    response = agent.chat(model.chatbot_prompt.format(query_str=query))
    return str(response)


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
    generate_responses(
        "How do I help my friend who is suffering from severe anxiety and depression?"
    )
