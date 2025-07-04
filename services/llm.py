import ast
import re
from fastapi import HTTPException
from typing import List
from sentence_transformers import SentenceTransformer

from services.db import connect
from services.groq import llm_connect
from models.llmModel import ContextOut

model = SentenceTransformer("all-mpnet-base-v2")


def transform_vector(vector: str) -> List[float]:
    return model.encode([vector])[0]


async def retrieve_context(query: str) -> List[ContextOut]:
    contexts: List[ContextOut] = []
    query_vector = transform_vector(query)

    try:
        client = connect()
        results = client.search(
            collection_name="rag_queries", query_vector=query_vector.tolist(), limit=2
        )

        for result in results:
            contexts.append(
                ContextOut(id=result.id, score=result.score, payload=result.payload)
            )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro ao recuperar contexto: {str(e)}"
        )

    return contexts


async def get_sql_query_with_database_structure(
    database_structure: str, order: str
) -> str:
    client = llm_connect()

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Você é um assistente especializado em SQL. "
                        f"Com base na seguinte estrutura de banco de dados:\n\n{database_structure}\n\n"
                        f"Gere uma única query SQL que atenda exatamente à seguinte solicitação do usuário. "
                        f"Retorne apenas a query SQL, sem explicações ou texto adicional."
                    ),
                },
                {
                    "role": "user",
                    "content": order,
                },
            ],
            model="gemma2-9b-it",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro no processamento da query com LLM: {str(e)}"
        )


async def get_result_interpretation(result: str, order: str) -> str:
    client = llm_connect()

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Você é um assistente especializado em SQL. "
                        f"Com base na seguinte pergunta do usuario: {order}"
                        f"Formate a resposta gerada pelo banco ao realizar a query de uma forma em linguagem natural."
                        f"Retorne apenas a resposta para o usuario, sem explicação."
                    ),
                },
                {
                    "role": "user",
                    "content": result,
                },
            ],
            model="gemma2-9b-it",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro no processamento da query com LLM: {str(e)}"
        )


async def optimize_generate(query: str, database_structure: str) -> str:
    client = llm_connect()

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "## Assistente Especializado em SQL e Otimização de Queries\n\n"
                        "Você é um **assistente especializado em SQL** e **otimização de desempenho de queries**. A seguir, você receberá:\n"
                        "* A **estrutura do banco de dados**\n"
                        "* Uma **query SQL** que precisa ser otimizada\n"
                        "---\n"
                        "### Tarefa\n"
                        "1. **Analisar** a query fornecida com base na estrutura do banco\n"
                        "2. **Sugerir comandos de otimização**, como criação de índices (`CREATE INDEX`), se necessário\n"
                        "3. **Reescrever a query** de forma mais eficiente, mantendo o mesmo resultado\n"
                        "---\n"
                        "### Formato da Resposta\n"
                        "* Retorne **apenas um JSON** com um **array de strings**, **sem explicações**\n"
                        "* O array deve conter, na ordem:\n"
                        "  1. **Comandos `CREATE INDEX`** (caso necessário)\n"
                        "  2. A **query otimizada**\n"
                        "#### Exemplos\n"
                        "**Com índices:**\n"
                        "[\n"
                        '  "CREATE INDEX idx_cliente_id ON pedidos(cliente_id);",\n'
                        '  "SELECT * FROM pedidos WHERE cliente_id = 123;"\n'
                        "]\n"
                        "**Sem necessidade de índices:**\n"
                        "[\n"
                        "  \"SELECT * FROM pedidos WHERE name = 'name';\"\n"
                        "]\n"
                        "---\n"
                        "### Estrutura do Banco de Dados\n"
                        f"{database_structure}\n"
                        "---\n"
                        "### Observações\n"
                        "* **Não inclua nenhuma explicação na resposta**\n"
                        "* Apenas o JSON com os comandos SQL e a query otimizada, conforme os exemplos acima\n"
                        "* Não retorne nenhum demarcador de bloco de código markdown\n"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query original:\n{query}",
                },
            ],
            model="gemma2-9b-it",
        )
        response_str = chat_completion.choices[0].message.content

        response_str = re.sub(r"\\'", "'", response_str)
        response_str = response_str.replace("‘", "'").replace("’", "'")  # aspas unicode
        response_str = response_str.replace("“", '"').replace(
            "”", '"'
        )  # aspas duplas unicode

        try:
            response_list = ast.literal_eval(response_str)
            for item in response_list:
                print(item)
        except (SyntaxError, ValueError):
            print("A resposta do LLM não pôde ser convertida para uma lista Python.")
        return response_list
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro no processamento da query com LLM: {str(e)}"
        )


async def create_database(database_structure: str) -> str:
    client = llm_connect()

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "- Você é um assistente especializado em bancos de dados relacionais. "
                        "- Sua tarefa é converter uma descrição de estrutura de banco de dados em comandos SQL do tipo DDL (Data Definition Language), como CREATE TABLE.\n\n"
                        "- Considere tipos de dados apropriados, chaves primárias, estrangeiras e restrições se estiverem descritas.\n\n"
                        "- Retorne **apenas os comandos SQL necessários** para criar as tabelas e relacionamentos descritos em ordem para criar.\n\n"
                        "- Importante: Retorne na ordem de criação correta.\n"
                        "- Exemplo de resposta esperada:\n"
                        "CREATE TABLE clientes (\n"
                        "    id INT PRIMARY KEY,\n"
                        "    nome VARCHAR(255),\n"
                        "    email VARCHAR(255) UNIQUE\n"
                        ");\n\n"
                        "CREATE TABLE pedidos (\n"
                        "    id INT PRIMARY KEY,\n"
                        "    cliente_id INT,\n"
                        "    data DATE,\n"
                        "    FOREIGN KEY (cliente_id) REFERENCES clientes(id)\n"
                        ");"
                    ),
                },
                {
                    "role": "user",
                    "content": database_structure,
                },
            ],
            model="gemma2-9b-it",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro no processamento da estrutura do banco com LLM: {str(e)}",
        )


async def populate_database(creation_command: str, number_insertions) -> str:
    client = llm_connect()

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente especializado em bancos de dados relacionais.\n"
                        "Sua tarefa é gerar comandos SQL de exemplo para popular um banco de dados já estruturado.\n\n"
                        "Com base nos comandos SQL de criação de tabelas fornecidos (CREATE TABLE), gere comandos INSERT INTO para povoar as tabelas com dados fictícios e coerentes.\n"
                        f"- Gere até {number_insertions} inserções por tabela.\n"
                        "- Os dados devem ser consistentes com os tipos definidos (ex: datas no formato YYYY-MM-DD, nomes fictícios, e-mails realistas, IDs coerentes).\n"
                        "- Considere os relacionamentos entre tabelas (ex: chaves estrangeiras devem apontar para registros válidos).\n\n"
                        "Retorne apenas os comandos INSERT INTO, mas **em formato de lista Python de strings**, por exemplo:\n"
                        "[\n"
                        "    \"INSERT INTO clientes (id, nome, email) VALUES (1, 'Francisco Silva', 'francisco@email.com');\",\n"
                        "    \"INSERT INTO clientes (id, nome, email) VALUES (2, 'Laura Lima', 'laura@email.com');\",\n"
                        "    ...\n"
                        "]"
                    ),
                },
                {
                    "role": "user",
                    "content": creation_command,
                },
            ],
            model="gemma2-9b-it",
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro no processamento da estrutura do banco com LLM: {str(e)}",
        )


def analyze_optimization_effects(
    original_metrics: dict,
    optimized_metrics: dict,
    original_query: str,
    optimized_query: str,
    applied_indexes: list,
) -> str:
    client = llm_connect()

    try:
        system_prompt = (
            "Você é um especialista em performance de banco de dados. "
            "Abaixo estão os resultados de execução de duas queries (original e otimizada) e os índices aplicados. "
            "Sua tarefa é avaliar se as otimizações devem ser mantidas com base nos resultados, "
            "considerando tempo, planos de execução e eficiência.\n\n"
            "Diga de forma clara e objetiva se vale a pena manter as mudanças. "
            "Considere possíveis riscos, ganhos marginais, impacto em outros tipos de consulta, e explique a decisão.\n\n"
            "Responda com uma análise técnica e objetiva."
        )

        user_prompt = (
            f"Query original:\n{original_query}\n\n"
            f"Métricas da query original:\n{original_metrics}\n\n"
            f"Query otimizada:\n{optimized_query}\n\n"
            f"Métricas da query otimizada:\n{optimized_metrics}\n\n"
            f"Índices aplicados:\n{applied_indexes}"
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="gemma2-9b-it",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao avaliar efeitos da otimização com LLM: {str(e)}",
        )
