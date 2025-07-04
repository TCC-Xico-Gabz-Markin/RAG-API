from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import llmRoutes
from routes import optimizerRoutes

app = FastAPI()

app.include_router(llmRoutes.router)
app.include_router(optimizerRoutes.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}
