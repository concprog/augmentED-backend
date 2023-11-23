import index
import model
from common import *

def create_subjectwise_indexes():
    indexes = {}
    for subject in subjects.keys():
        indexes[subject] = index.PersistentDocStoreFaiss(data_path=get_subject_data_path(subject),storage_path=r"vectorstores/{}".format(subjects[subject]), service_context=model.g_service_ctx).load_or_create()

    return indexes

def create_subjectwise_tools(indexes):
    tools = {}
    for subject in indexes:
        tools[subject] = model.subject_vector_tool(indexes[subject], subject)
    return tools

def create_chat_agent(llm=model.load_llm(MODEL_PATH), tools=[], from_dict=False):
    tools = list(tools.values) if from_dict else tools
    return model.response_agent(llm=llm, tools=tools)

indexes = create_subjectwise_indexes()
tools = create_subjectwise_tools(indexes)
agent = create_chat_agent(llm = model.load_llm(MODEL_PATH), tools=tools, from_dict=True)

# Accessible fn.s

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
    return []

if __name__ == "__main__":
    generate_responses("How do I help my friend who is suffering from severe anxiety and depression?")