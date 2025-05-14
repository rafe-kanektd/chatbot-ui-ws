from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.requests import QueryRequest
from agent.agent import query_llm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
async def search(query: QueryRequest):
    response = query_llm(query.user_input, model=query.model)
    return {"result": response}