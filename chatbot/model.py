from dataclasses import dataclass, field
from typing import Generator, List, Set, Dict, Optional, Tuple, Union
from os.path import sep as PathSep

from llama_index import ServiceContext, Prompt
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)


DATA_PATH = "data/"
DB_FAISS_PATH = "chatbot/vectorstore/db_faiss"

MODEL_PATH = "chatbot/models/nous-hermes-llama-2-7b.Q5_K_M.gguf"
MODEL_NAME = MODEL_PATH.partition(".")[0].partition(PathSep)[-1]
MODEL_KWARGS = {"max_new_tokens": 1024, "temperature": 0.8}

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_ARGS = model_kwargs = {"device": "cuda"}

SIMILARITY_SEARCH_KWARGS = {"k": 3}
SIMILARITY_SEARCH_THRESHOLD = 0.33

instruct_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input: 
{input}

### Response:

"""
# TODO Do prompt engineering to fix the instruction and other stuff
chatbot_instruction = "" 
instruct_prompt = Prompt(instruct_prompt_template.format(instruction=chatbot_instruction))


def set_custom_prompt(
    custom_prompt_template, input_variables=["query_str", "context_str"]
):
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = Prompt(custom_prompt_template)
    return prompt


# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = LlamaCPP(
        model=MODEL_PATH,
        model_type="llama",
        max_new_tokens=4096,
        temperature=0.5,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 16},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    return llm


@dataclass
class chat_history:
    history: List[Dict[str, str]] = field(default_factory=list)
    roles: Set[str] = field(default_factory=set)

    def append_to_chat_history(self, message, role):
        if not role in self.roles:
            self.roles.add(role)
        self.history.append({role: message})

    def get_all_messages_by_role(self, role):
        messages = []
        if role in self.roles:
            messages = [chat[role] for chat in self.history]
        return messages

    def get_last_n_conversations(self, n=10):
        messages = self.history[:n]
        conversation = "\n".join([": ".join(*chat.items()) for chat in messages])
        return conversation

    def get_all_conversations(self):
        conversation = "\n".join([": ".join(*chat.items()) for chat in self.history])
        return conversation

    def get_as_tuples(self):
        conversation = [chat.items() for chat in self.history]
        return conversation


def instruct_qa_engine(prompt_instruction, prompt_input):
    base_ctx = ServiceContext(embed_model=embeddings, llm=llm)
    return base_engine


# Globals

embeddings = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL,
)

psych_db = FAISS.load_local(DB_FAISS_PATH, embeddings)
llm = load_llm()

chat_history = chat_history()

rag_qa_prompt = set_custom_prompt(
    custom_prompt_template=instruct_prompt_template.format(
        **{
            "instruction": therapist_prompt_instruction,
            "input": therapist_prompt_input,
        }
    )
)

def generate_response_with_rag(query, subject)
def generate_openai_response(message):
    choices = []

    choices.append(
        {
            "role": "assistant",
            "content": message,
        }
    )
    data = {"choices": choices}
    return data


if __name__ == "__main__":
    pass
