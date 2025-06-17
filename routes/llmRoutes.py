from fastapi import APIRouter, Depends, HTTPException

from dependencies import get_api_key
from services.llm import LLMService
from models.payloadRAG import RAGQueryRequest, RAGQueryResponse
from models.payloadInterpreter import InterpreterQueryRequest, InterpreterQueryResponse

router = APIRouter(prefix="/rag", tags=["Query"], dependencies=[Depends(get_api_key)])
service = LLMService()


@router.post("/query/structure", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    try:
        query = await service.get_sql_query_with_database_structure(
            database_structure=request.database_structure, order=request.order
        )
        return RAGQueryResponse(query=query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar RAG: {str(e)}")


@router.post("/query/interpreter", response_model=InterpreterQueryResponse)
async def query_interpreter(request: InterpreterQueryRequest):
    try:
        response = await service.get_result_interpretation(
            result=request.result, order=request.order
        )
        return InterpreterQueryResponse(response=response)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro ao processar Interpretador: {str(e)}"
        )
