from fastapi import APIRouter

from fastapi import HTTPException
from fastapi import UploadFile, File
from fastapi.responses import FileResponse

import os
from app import functions


from chatbot import functions as ai_functions

router = APIRouter()

DATA_STORE_PATH = "data/user_doc"


@router.post("/uploadbook/")
async def up_and_down(text: str, file: UploadFile = File(...)):
    file_path = functions.get_file_path(file.filename)
    if file.content_type != "application/pdf":
        raise HTTPException(400, detail=f"Invalid document type: {file_path}")
    else:
        data = file.file.read()
        new_fileName = "{}_{}.pdf".format(
            os.path.splitext(str(file.filename))[0], functions.timestr()
        )
        save_file_path = functions.get_file_path(os.path.join(DATA_STORE_PATH, new_fileName))
        print(save_file_path)
        with open(save_file_path, "wb") as f:
            f.write(data)
        ai_functions.set_document_chat_engine(save_file_path)
        return (
            FileResponse(
                path=save_file_path,
                media_type="application/octet-stream",
                filename=new_fileName,
            ),
            text,
            file_path,
        )


@router.post("/getbook/")
async def get_file(filename: str):
    return functions.serve_book(functions.get_flie_path_from_name(filename))



@router.post("/reindex/")
def reindex(passwd: str):
    if passwd != "recreate":
        raise HTTPException(400, detail=f"Invalid password: {passwd}")
    else:
        ai_functions.recreate_indexes(passwd)
        return {"status": "indexes recreated"}
        

@router.post("/documentchat/")
async def read_conversation(
    query: str,
):
    response = ai_functions.chat_with_document(query=query)
    return ai_functions.generate_text_from_response(response)

@router.post("/summarizer/")
async def summarize(
    para: str,
):
    response = ai_functions.plain_text_summarizer(para)
    return ai_functions.generate_text_from_response(response)

@router.post("/notemaker/")
async def make_notes(
    para: str,
):
    response = ai_functions.note_maker_summarize(para, n_paras=2)
    return ai_functions.generate_text_from_response(response)

@router.post("/similaritysearch/")
async def search_similar_para(
    para: str,
):
    responses = ai_functions.search_passages(passage=para, top_k=4)
    response = {"paragraphs": responses}
    return response

