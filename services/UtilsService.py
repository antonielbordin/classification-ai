import os
import re
import pytesseract
import shutil
import docx
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image

# 🔹 Caminho onde os arquivos serão processados
PROCESS_DIR = "process"
os.makedirs(PROCESS_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    """Limpa o texto OCRizado, removendo quebras de linha e espaços extras."""
    if not text:
        return ""
    text = re.sub(r'\n+', ' ', text)  # Substitui múltiplas quebras de linha por espaço
    text = re.sub(r'\s+', ' ', text).strip()  # Remove espaços duplicados
    return text

def extract_text_from_file(filepath: str) -> str:
    """Extrai texto de arquivos suportados: imagens, PDFs, DOCX, XLSX e TXT."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in [".png", ".jpg", ".jpeg"]:
            img = Image.open(filepath)
            return clean_text(pytesseract.image_to_string(img))

        elif ext == ".pdf":
            text = ""
            with fitz.open(filepath) as doc:
                for page in doc:
                    text += page.get_text()
            return clean_text(text)

        elif ext == ".docx":
            doc = docx.Document(filepath)
            return clean_text("\n".join(p.text for p in doc.paragraphs))

        elif ext == ".xlsx":
            df = pd.read_excel(filepath, header=None)
            return clean_text("\n".join(df.astype(str).values.flatten()))

        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return clean_text(f.read())

        else:
            print(f"❌ Tipo de arquivo não suportado: {filepath}")
            return ""

    except Exception as e:
        print(f"⚠️ Erro ao processar {filepath}: {e}")
        return ""

def clear_process_folder(folder_path: str = PROCESS_DIR):
    """Limpa todos os arquivos da pasta de processamento."""
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"⚠️ Erro ao limpar pasta de processamento: {e}")
