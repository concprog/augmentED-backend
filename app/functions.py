from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse
import time
import os
timestr = time.strftime("%Y%m%d-%H%M%S")
from fastapi import FastAPI, File,UploadFile
import uvicorn
from chatbot import functions as ai_functions

def serve_book(filepath: str):
    try:
        return FileResponse(filepath, media_type='application/pdf', filename='book_file.pdf')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))


app = FastAPI()


base_dir = os.path.dirname(os.path.abspath(__file__))
upload_dir = os.path.join(base_dir,"data")

@app.get("/")
def reading():
    return "FastAPI backend for augmentED"

@app.get("/filepath")
def get_file_path(file_name):
    file_path = os.path.abspath(file_name)  # Join the directory and file name
    return file_path



@app.post("/file/uploadanddownload")

def up_and_down(text: str, file: UploadFile = File(...)):
    file_path = get_file_path(file.filename)
    if file.content_type != "application/pdf":
        raise HTTPException(400, detail=f"Invalid document type: {file_path}")
    else:
        data = file.file.read()
        new_fileName = "{}_{}.pdf".format(os.path.splitext(str(file.filename))[0],timestr)
        save_file_path = os.path.join(upload_dir,new_fileName)
        with open(save_file_path, "wb") as f:
            f.write(data)
        ai_functions.set_document_chat_engine(file_path)
        return FileResponse(path=save_file_path,media_type="application/octet-stream",filename=new_fileName), text,file_path
    


if __name__== '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)