import logging
import os

def get_logger(name="cf-monitoring"):

    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler("logs/pipeline.log")
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
