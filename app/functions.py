from fastapi.responses import FileResponse
from fastapi import HTTPException
import os
import time

def serve_book(filepath: str):
    try:
        return FileResponse(filepath, media_type='application/pdf', filename='book_file.pdf')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))


timestr = time.strftime("%Y%m%d-%H%M%S")


def get_file_path(file_name):
    file_path = os.path.abspath(file_name)  # Join the directory and file name
    return file_path



    