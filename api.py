"""
Serve the query pipeline over HTTP so a frontend / other services can call it.

    pip install fastapi uvicorn
    export GEMINI_API_KEY=...
    uvicorn api:app --reload --port 8000

    curl -X POST http://localhost:8000/query \
        -H "Authorization: Bearer demo-token-engineer" \
        -H "Content-Type: application/json" \
        -d '{"query": "What is our refund policy?"}'
"""

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from query.pipeline import QueryPipeline
from query.auth import AuthenticationError
from logging_setup import setup_logging

setup_logging()
app = FastAPI(title="RAG Query API")
pipeline = QueryPipeline()  # loaded once; reused across requests


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None


class SourceChunk(BaseModel):
    source: str
    page: int | None = None
    section: str | None = None
    sensitivity_tier: str


class QueryResponse(BaseModel):
    answer: str
    refined_query: str
    sources: list[SourceChunk]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, authorization: str = Header(...)):
    token = authorization.removeprefix("Bearer ").strip()

    if req.top_k:
        pipeline.top_k = req.top_k

    try:
        result = pipeline.handle_query(auth_token=token, query=req.query)
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid or expired credential.")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    return QueryResponse(
        answer=result.answer,
        refined_query=result.refined_query,
        sources=[
            SourceChunk(
                source=h["metadata"]["source"],
                page=h["metadata"].get("page"),
                section=h["metadata"].get("section"),
                sensitivity_tier=h["metadata"]["sensitivity_tier"],
            )
            for h in result.source_chunks
        ],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
