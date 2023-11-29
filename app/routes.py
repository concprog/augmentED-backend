

from fastapi import APIRouter
from fastapi import UploadFile


from app import models, functions


from chatbot.functions import generate_generic_response, generate_openai_from_response, search_catalogue


router = APIRouter()

DATA_PATH="data/"

@router.post("/uploadbook/")
async def create_upload_file(file: UploadFile | None = None, subject: str | None = None):
    if not file:
        return {"message": "No upload file sent"}
    else:
        book_content = await file.read()
        # TODO add code to save it in sql table
        return {"filename": file.filename}

# Psuedocode - Replace with working example
@router.post("/getbook/")
async def get_file(filename: str):
    return {"file": functions.serve_book(DATA_PATH+filename)}

@router.post("/conversation/")
async def read_conversation(
    query: str,
):
    # TODO Add subject/model checking and redirection

    response = generate_generic_response(query=query)
    return generate_openai_from_response(response)

router.post("/para_similarity_search/")
async def search_similar_para(
    para: str,
):
    # TODO Add subject/model checking and redirection

    response = search_catalogue(query=para, top_k=4)
    return generate_openai_from_response(response)