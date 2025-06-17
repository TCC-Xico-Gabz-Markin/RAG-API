import ast
import re

from loguru import logger


def process_llm_output(output: str):
    response_str = re.sub(r"\\'", "'", output)
    response_str = response_str.replace("‘", "'").replace("’", "'")  # aspas unicode
    response_str = response_str.replace("“", '"').replace(
        "”", '"'
    )  # aspas duplas unicode

    try:
        response_list = ast.literal_eval(response_str)
        return response_list
    except (SyntaxError, ValueError) as e:
        logger.warning(
            "A resposta do LLM não pôde ser convertida para uma lista Python."
        )
        raise ValueError(f"Falha ao processar a saída do LLM: {e}") from e
