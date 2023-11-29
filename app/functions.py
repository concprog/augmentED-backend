from fastapi.responses import StreamingResponse
from fastapi import Depends, HTTPException

def serve_book(filepath: str):
    try:
        file_stream = open(filepath, mode="rb")
        return StreamingResponse(file_stream, media_type='application/pdf')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def save_book(filename):
    pass