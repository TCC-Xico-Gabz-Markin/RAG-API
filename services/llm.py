from fastapi import HTTPException
from typing import List
from sentence_transformers import SentenceTransformer

from helpers.helpers import process_llm_output
from services.groq import llm_connect
from models.llmModel import ContextOut


class LLMService:
    def __init__(self):
        self.client = llm_connect()
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.attempt = 0

    def transform_vector(self, vector: str) -> List[float]:
        return self.model.encode([vector])[0]

    async def retrieve_context(self, query: str) -> List[ContextOut]:
        contexts: List[ContextOut] = []
        query_vector = self.transform_vector(query)

        try:
            results = self.client.search(
                collection_name="rag_queries",
                query_vector=query_vector.tolist(),
                limit=2,
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
        self, database_structure: str, order: str
    ) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
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
                status_code=500,
                detail=f"Erro no processamento da query com LLM: {str(e)}",
            )

    async def get_result_interpretation(self, result: str, order: str) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
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
                status_code=500,
                detail=f"Erro no processamento da query com LLM: {str(e)}",
            )

    async def optimize_generate(self, query: str, database_structure: str) -> str:
        chat_completion = self.client.chat.completions.create(
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
                        "* Não retorne nenhum demarcador de bloco de código markdown como ```json ou ```sql\n"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query original:\n{query}",
                },
            ],
            model="gemma2-9b-it",
        )
        try:
            if self.attempt >= 3:
                self.attempt = 0
            response = process_llm_output(chat_completion.choices[0].message.content)
        except Exception as e:
            self.attempt += 1
            if self.attempt < 3:
                return await self.optimize_generate(query, database_structure)
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao processar a resposta do LLM após 3 tentativas: {str(e)}",
            )
        return response

    async def create_database(self, database_structure: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "## Assistente Especializado em SQL e Otimização de Queries\n\n"
                        "Você é um **assistente especializado em SQL** e **criação de bancos de dados relacionais**. A seguir, você receberá:\n"
                        "* A **estrutura do banco de dados**\n"
                        "---\n"
                        "### Tarefa\n"
                        "**Converter** uma descrição de estrutura de banco de dados em comandos SQL do tipo DDL (Data Definition Language), como CREATE TABLEuma descrição de estrutura de banco de dados em comandos SQL do tipo DDL (Data Definition Language).\n"
                        "---\n"
                        "### Formato da Resposta\n"
                        "* Retorne **apenas um JSON** com um **array de strings**, **sem explicações**\n"
                        "* O array deve conter **apenas os comandos SQL necessários** para criar as tabelas e relacionamentos descritos em ordem para criar.\n"
                        "- Considere tipos de dados apropriados, chaves primárias, estrangeiras e restrições se estiverem descritas.\n\n"
                        "- Importante: Retorne na ordem de criação correta.\n"
                        "---\n"
                        "#### Exemplos:\n"
                        "CREATE TABLE clientes (id INT PRIMARY KEY, nome VARCHAR(255), email VARCHAR(255) UNIQUE);\n"
                        "CREATE TABLE pedidos (id INT PRIMARY KEY, cliente_id INT, data DATE, FOREIGN KEY (cliente_id) REFERENCES clientes(id));\n"
                        "---\n"
                        "### Estrutura do Banco de Dados\n"
                        f"{database_structure}\n"
                        "---\n"
                        "### Observações\n"
                        "* **Não inclua nenhuma explicação na resposta**\n"
                        "* Apenas a lista de strings com os comandos SQL do tipo DDL, conforme os exemplos acima\n"
                        "* Não retorne nenhum demarcador de bloco de código markdown como ```json ou ```sql\n"
                    ),
                },
                {
                    "role": "user",
                    "content": database_structure,
                },
            ],
            model="gemma2-9b-it",
        )
        try:
            if self.attempt >= 5:
                self.attempt = 0
            response = process_llm_output(chat_completion.choices[0].message.content)
        except Exception as e:
            self.attempt += 1
            if self.attempt < 5:
                return await self.create_database(database_structure)
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao processar a resposta do LLM após {self.attempt} tentativas: {str(e)}",
            )
        return response

    async def populate_database(self, creation_command: str, number_insertions) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
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
        self,
        original_metrics: dict,
        optimized_metrics: dict,
        original_query: str,
        optimized_query: str,
        applied_indexes: list,
    ) -> str:
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

            chat_completion = self.client.chat.completions.create(
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
