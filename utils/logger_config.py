import logging
import os

def get_logger(name: str = __name__) -> logging.Logger:
    """
    Retorna um logger configurado com saída para arquivo e terminal.

    - Cria o diretório de logs se necessário.
    - Adiciona handlers apenas uma vez por logger.
    """

    # Caminho seguro para ambientes como Render: /tmp/logs ou ./logs
    base_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    
    try:
        os.makedirs(base_log_dir, exist_ok=True)
    except PermissionError:
        # Fallback para ambiente temporário se não conseguir criar a pasta desejada
        base_log_dir = "/tmp/logs"
        os.makedirs(base_log_dir, exist_ok=True)

    log_file = os.path.join(base_log_dir, "app.log")

    logger = logging.getLogger(name)

    # Evita adicionar múltiplos handlers em execuções repetidas
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
