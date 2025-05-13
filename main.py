from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import shutil

from utils.logger_config import get_logger
from services.UtilsService import extract_text_from_file, clean_text, clear_process_folder
from services.ClassifyDocsService import classify_document

logger = get_logger(__name__)

PROCESS_FOLDER = "process"

os.makedirs(PROCESS_FOLDER, exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("homepage.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_and_classify(request: Request, file: UploadFile = File(...)):
    try:
        file_path = os.path.join(PROCESS_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = extract_text_from_file(file_path)
        if not text:
            logger.warning(f"Nenhum texto extraído de {file.filename}.")
            return templates.TemplateResponse("homepage.html", {
                "request": request,
                "result": {"filename": file.filename, "classification": "Texto não extraído"}
            })

        text = clean_text(text)
        document_type, confidence = classify_document(text)

        return templates.TemplateResponse("homepage.html", {
            "request": request,
            "result": {
                "filename": file.filename,
                "classification": document_type,
                "confidence": f"{confidence:.2%}"
            }
        })

    except Exception as e:
        logger.error(f"Erro ao classificar documento {file.filename}: {e}")
        return templates.TemplateResponse("homepage.html", {
            "request": request,
            "result": {
                "filename": file.filename,
                "classification": f"Erro interno: {str(e)}"
            }
        })

    finally:
        clear_process_folder()
