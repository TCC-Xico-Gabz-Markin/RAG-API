from pydantic import BaseModel
from typing import List, Dict


class OptimizerRequest(BaseModel):
    query: str
    database_structure: str


class OptimizerResponse(BaseModel):
    result: List[str]


class CreateDatabaseRequest(BaseModel):
    database_structure: str


class CreateDatabaseResponse(BaseModel):
    sql: List[str]


class PopulateDatabaseResponse(BaseModel):
    sql: str


class PopulateDatabaseRequest(BaseModel):
    creation_command: str
    number_insertions: int


class OptimizationAnalysisRequest(BaseModel):
    original_metrics: Dict
    optimized_metrics: Dict
    original_query: str
    optimized_query: str
    applied_indexes: List[str]


class OptimizationAnalysisResponse(BaseModel):
    analysis: str
