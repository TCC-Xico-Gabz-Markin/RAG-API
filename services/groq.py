from groq import Groq
from dotenv import load_dotenv
import os

from loguru import logger

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


def llm_connect():
    try:
        client = Groq(api_key=api_key)
        if client:
            logger.info("Conexão com o Groq estabelecida com sucesso.")
        return client
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        raise e
