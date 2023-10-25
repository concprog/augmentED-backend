from dataclasses import dataclass, field
from typing import Generator, List, Set, Dict, Optional, Tuple, Union
from os.path import sep as PathSep

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import (
    LLMChain,
    ConversationalRetrievalChain,
)
from langchain.memory import ConversationBufferWindowMemory



DATA_PATH = "data/"
DB_FAISS_PATH = "chatbot/vectorstore/db_faiss"

MODEL_PATH = "chatbot/models/nous-hermes-llama-2-7b.Q5_K_M.gguf"
MODEL_NAME = MODEL_PATH.partition(".")[0].partition(PathSep)[-1]
MODEL_KWARGS = {"max_new_tokens": 1024, "temperature": 0.8}

EMBEDDING_MODEL = "thenlper/gte-small"
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

therapist_prompt_instruction = """You are a responsive and skilled therapist taking care of a patient who is looking for guidance and advice on managing their emotions, stress, anxiety and other mental health issues through chat based therapy. Attentively listen to the patient and answer the patient's questions in an empathetic and non-judgemental tone, and do not judge the patient for any issues they are facing. Offer acceptance, support, and care for the patient, regardless of their circumstances or struggles.  Make them comfortable and ask open ended questions in an empathetic manner that encourages self reflection. Listen to the patient's problem and provide them with an outlet for their issues. Try to further understand the patient's problem and help them solve their problems if and only if they want you to solve it. Also try to avoid giving false or misleading information, and caveat when you entirely sure about the right answer.
Additionally, use the following context to chat with the user:
{chat_history}
Respond to the patient and give them a concise response, asking open questions that encourage reflection, self-introspection and deep thought.
"""

therapist_prompt_input = "{question}"



def set_custom_prompt(
    custom_prompt_template, input_variables=["question", "chat_history"]
):
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=input_variables
    )
    return prompt


# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        max_new_tokens=4096,
        temperature=0.75,
        gpu_layers=16,
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


def conv_retr_chain(llm, prompt, db, lexical_retriever=None):
    # retriever = db.as_retriever(search_kwargs={'k': 3})
    # if lexical_retriever != None:
    #     retriever = EnsembleRetriever(retrievers = [db.as_retriever(search_kwargs = {"k" :2 }), lexical_retriever])
    conv_retr_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
    )
    return conv_retr_chain


def instruct_chain(prompt_instruction, prompt_input):
    base_chain = LLMChain(
        llm=llm,
        prompt=set_custom_prompt(
            instruct_prompt_template.format(
                **{"instruction": prompt_instruction, "input": prompt_input}
            ),
            input_variables=["question"],
        ),
    )
    return base_chain


# Globals

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL, model_kwargs=EMBEDDING_MODEL_ARGS
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
retrieval_chatbot_chain = conv_retr_chain(llm, rag_qa_prompt, psych_db)

# QA Model Function


def generate_therapist_response_with_rag(query):
    

    while True:
        if query == "exit" or query == "quit" or query == "q" or query == "f":
            print("Exiting")
        if query == "":
            continue
        result = retrieval_chatbot_chain(
            {"question": query, "chat_history": chat_history.history}
        )
        response = result["answer"]

        return response


def generate_therapist_response_without_rag(query, roles=["Patient", "Therapist"]):
    if chat_history.history == []:
        chat_history.append_to_chat_history(
            "Hi, is there anything you want to talk to me about?", roles[1]
        )
    fmted_chat_history = chat_history.get_all_conversations()
    response_chain = instruct_chain(
        therapist_prompt_instruction.format(chat_history=fmted_chat_history),
        therapist_prompt_input,
    )
    response = response_chain.run(query)
    chat_history.append_to_chat_history(query, roles[0])
    chat_history.append_to_chat_history(response, roles[1])
    return response


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
    print(
        generate_therapist_response_without_rag(
        instruct_prompt_template.format(instruction = therapist_prompt_instruction, input=therapist_prompt_input.format(question="What is the meaning of life? What do you think it could be, after all?"))
        )
    )
