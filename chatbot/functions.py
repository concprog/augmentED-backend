from llama_index import set_global_service_context

from chatbot import model


set_global_service_context(model.g_service_ctx)


document_chat_engine_cache = []


# Accessible fn.s

# Create/Recreate

def recreate_indexes(passwd: str):
    if passwd == "recreate":
        model.everything_pipeline.run_pipeline(create=True)

# Responses
def generate_generic_response(query):
    response = model.agent.chat(model.chatbot_prompt.format(query_str=query))
    return str(response)


def set_document_chat_engine(file_path):
    doc_chat_engine = model.everything_pipeline.query_one_file(file_path=file_path)
    if len(document_chat_engine_cache) > 1:
        document_chat_engine_cache.pop()
    document_chat_engine_cache.append(doc_chat_engine)


def chat_with_document(query):
    doc_chat_engine = document_chat_engine_cache[0]
    resp = doc_chat_engine.query(query)
    return str(resp)


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

def generate_text_from_response(response):
    data = {"text": response}
    return data

# Summarizers

def plain_text_summarizer(text):
    return model.summarize_text(text)

def note_maker_summarize(text, n_paras: int = 2):
    return model.augmented_summarize(text, n_paras)

# Search

def search_passages(passage, top_k=3):
    return model.search_for_paras(passage, top_k=top_k)

def search_titles(query, top_k=5):
    return model.search_for_title(query, top_k=top_k)


if __name__ == "__main__":
    set_document_chat_engine("chatbot/data/psychology/lehe108.pdf")
    print(chat_with_document("what is this book about?"))
