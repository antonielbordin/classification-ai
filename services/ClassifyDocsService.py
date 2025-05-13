import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import BigBirdTokenizerFast
from scipy.special import softmax

# Configuração de logging
from utils.logger_config import get_logger
logger = get_logger(__name__)

# 🔹 Diretório base absoluto
BASE_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = (BASE_DIR / ".." / "models" / "bigbird_trained").resolve()
MODEL_PATH = (BASE_DIR / ".." / "models" / "bigbird_trained.onnx").resolve()

logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"TOKENIZER_PATH: {TOKENIZER_PATH}")
logger.info(f"MODEL_PATH: {MODEL_PATH}")

# 🔹 Carregar tokenizador e modelo ONNX
logger.info(f"📦 Carregando tokenizador de: {TOKENIZER_PATH}")
bigbird_tokenizer = BigBirdTokenizerFast.from_pretrained(
    str(TOKENIZER_PATH),
    local_files_only=True)

logger.info(f"🧠 Carregando modelo ONNX de: {MODEL_PATH}")
onnx_bigbird = ort.InferenceSession(str(MODEL_PATH))

# 🔹 Rótulos usados no treinamento
LABELS = [
    "ATA DE REUNIAO", "PROPOSTA COMERCIAL", "COMPROVANTE DE PAGAMENTO",
    "CONTRATO", "MANUAL", "ATESTADO DE SAUDE OCUPACIONAL", "ORDEM DE COMPRA",
    "FICHA CADASTRAL", "ACEITE", "NOTA DE EMPENHO", "CERTIDÃO NEGATIVA", 
    "DESCONHECIDO"
]

logger.info(f"📥 Entradas esperadas pelo modelo: {[i.name for i in onnx_bigbird.get_inputs()]}")

def classify_document(text):
    logger.info("📝 Iniciando classificação de documento...")
    start_total = time.perf_counter()

    if not text or not isinstance(text, str):
        logger.warning("⚠️ Texto de entrada inválido ou vazio.")
        return "DESCONHECIDO", 0.0

    # 🔹 Tokenização
    start_token = time.perf_counter()
    inputs = bigbird_tokenizer(
        text, 
        return_tensors="np",
        max_length=704,
        truncation=True,
        padding="max_length"
    )
    end_token = time.perf_counter()
    logger.info(f"🔧 Tokenização concluída em {end_token - start_token:.4f}s")

    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.float32)

    # 🔹 Inferência com ONNX
    start_infer = time.perf_counter()
    outputs = onnx_bigbird.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    end_infer = time.perf_counter()
    logger.info(f"⚙️  Inferência concluída em {end_infer - start_infer:.4f}s")
    
    # 🔹 Classificação com probabilidade
    logits = outputs[0][0]
    probs = softmax(logits)
    predicted_label = int(np.argmax(probs))
    confidence = float(probs[predicted_label])
    label = LABELS[predicted_label] if predicted_label < len(LABELS) else "DESCONHECIDO"

    total_time = time.perf_counter() - start_total
    logger.info(f"✅ Documento classificado como: {label} (confiança: {confidence:.2%})")
    logger.info(f"⏱️ Tempo total: {total_time:.4f}s\n")

    return label, confidence