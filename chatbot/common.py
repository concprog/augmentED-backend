from os.path import sep as PathSep
import os


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
EMBEDDING_MODEL_ARGS = model_kwargs = {"device": "cuda"}

DATA_PATH = os.curdir + PathSep + "data/"
DB_FAISS_PATH = "chatbot/vectorstore/db_faiss"

MODEL_PATH = os.getcwd()+PathSep+"chatbot/models/nous-hermes-llama-2-7Q5_K_M.gguf"
MODEL_NAME = MODEL_PATH.partition(".")[0].partition(PathSep)[-1]
MODEL_KWARGS = {"max_new_tokens": 1024, "temperature": 0.8}

SIMILARITY_SEARCH_KWARGS = {"k": 3}
SIMILARITY_SEARCH_THRESHOLD = 0.33

instruct_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input: 
{input}

### Response:

"""

# library commons

subjects = {"maths": "maths", "eee": "eee", "psychology": "psychology"}


def get_subject_data_path(subject_str: str) -> str:
    if subject_str in subjects.keys():
        return DATA_PATH + PathSep + subjects[subject_str]
    return None
