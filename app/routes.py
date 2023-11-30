

from fastapi import APIRouter

from fastapi import HTTPException
from fastapi import UploadFile, File
from fastapi.responses import FileResponse

import os
from app import functions


from chatbot import functions as ai_functions

router = APIRouter()

DATA_PATH="data/"

@router.post("/uploadbook/")
def up_and_down(text: str, file: UploadFile = File(...)):
    file_path = functions.get_file_path(file.filename)
    if file.content_type != "application/pdf":
        raise HTTPException(400, detail=f"Invalid document type: {file_path}")
    else:
        data = file.file.read()
        new_fileName = "{}_{}.pdf".format(os.path.splitext(str(file.filename))[0],functions.timestr)
        save_file_path = os.path.join(DATA_PATH,new_fileName)
        with open(save_file_path, "wb") as f:
            f.write(data)
        ai_functions.set_document_chat_engine(file_path)
        return FileResponse(path=save_file_path,media_type="application/octet-stream",filename=new_fileName), text,file_path

# Psuedocode - Replace with working example
@router.post("/getbook/")
async def get_file(filename: str):
    return {"file": functions.serve_book(DATA_PATH+filename)}

@router.post("/conversation/")
async def read_conversation(
    query: str,
):
    # TODO Add subject/model checking and redirection

    response = ai_functions.generate_generic_response(query=query)
    return ai_functions.generate_openai_from_response(response)

router.post("/para_similarity_search/")
async def search_similar_para(
    para: str,
):
    # TODO Add subject/model checking and redirection

    response = ai_functions.search_catalogue(query=para, top_k=4)
    return ai_functions.generate_openai_from_response(response)