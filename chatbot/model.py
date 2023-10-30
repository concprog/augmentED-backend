from dataclasses import dataclass, field
from typing import Generator, List, Set, Dict, Optional, Tuple, Union
from os.path import sep as PathSep

from llama_index import Prompt, ServiceContext, set_global_service_context
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

import index

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
chatbot_instruction = "Solve the problems given below to the best of your ability. Remember, for each wrong answer marks are deducted, hence answer carefully and leave the answer blank and caveat when you are not sure of your solution.\nUse the following notes to anwer the question: {context_str}" 
instruct_prompt = Prompt(instruct_prompt_template.format(instruction=chatbot_instruction, input="{input}"))


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
        model_path=MODEL_PATH,
        max_new_tokens=3900,
        temperature=0.25,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 18},
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





# Globals



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
    llm = load_llm()
    embeddings = HuggingFaceEmbedding(EMBEDDING_MODEL)
    g_service_ctx = ServiceContext.from_defaults(llm=llm, embed_model=embeddings)
    set_global_service_context(g_service_ctx)
    vsi = index.PersistentDocStoreFaiss().load_or_create_default()
    print(vsi.as_query_engine().query("I feel overwhelmed by life. How can I fix this?"))
