from fastapi import APIRouter, Depends, HTTPException
from dependencies import get_api_key
from services.llm import LLMService
from models.payloadOptimizer import (
    OptimizerRequest,
    OptimizerResponse,
    CreateDatabaseRequest,
    CreateDatabaseResponse,
    PopulateDatabaseResponse,
    PopulateDatabaseRequest,
    OptimizationAnalysisRequest,
    OptimizationAnalysisResponse,
)

router = APIRouter(
    prefix="/optimizer", tags=["Optimizer"], dependencies=[Depends(get_api_key)]
)
service = LLMService()


@router.post("/generate", response_model=OptimizerResponse)
async def optimize_query(request: OptimizerRequest):
    try:
        result = await service.optimize_generate(
            query=request.query,
            database_structure=request.database_structure,
        )
        return OptimizerResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-database", response_model=CreateDatabaseResponse)
async def create_db(request: CreateDatabaseRequest):
    try:
        sql = await service.create_database(
            database_structure=request.database_structure
        )
        return CreateDatabaseResponse(sql=sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/populate", response_model=PopulateDatabaseResponse)
async def populate_db(request: PopulateDatabaseRequest):
    try:
        sql = await service.populate_database(
            creation_command=request.creation_command,
            number_insertions=request.number_insertions,
        )
        return PopulateDatabaseResponse(sql=sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=OptimizationAnalysisResponse)
async def analyze(request: OptimizationAnalysisRequest):
    try:
        result = service.analyze_optimization_effects(
            original_metrics=request.original_metrics,
            optimized_metrics=request.optimized_metrics,
            original_query=request.original_query,
            optimized_query=request.optimized_query,
            applied_indexes=request.applied_indexes,
        )
        return OptimizationAnalysisResponse(analysis=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
