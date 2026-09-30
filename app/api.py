"""REST API.

    uvicorn app.api:app --reload
    open http://localhost:8000/docs

FastAPI generates interactive docs from the Pydantic models, so /docs is a
working console you can demo without a frontend.
"""

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.graph import ask
from app.llm import describe
from app.schemas import PolicyAnswer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("policy-agent")

app = FastAPI(
    title="Policy Agent",
    description="Multi-step RAG agent over company policy documents.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class QueryResponse(BaseModel):
    request_id: str
    latency_seconds: float
    result: PolicyAnswer


@app.get("/health")
def health() -> dict:
    """Liveness check. Also reports the active provider, which saves a lot of
    confusion when something is misconfigured."""
    return {"status": "ok", "model": describe()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Answer a policy question.

    Every request gets an id and a logged latency. When something goes wrong
    in production, the first question is always "which request, and how long
    did it take" — so log it from day one.
    """
    request_id = uuid.uuid4().hex[:8]
    log.info("[%s] query: %s", request_id, req.question)

    start = time.perf_counter()
    try:
        result = ask(req.question)
    except Exception as exc:
        log.exception("[%s] failed", request_id)
        raise HTTPException(status_code=502, detail=f"Agent failed: {exc}")

    latency = time.perf_counter() - start
    log.info(
        "[%s] done in %.2fs — amount=%s supported=%s",
        request_id,
        latency,
        result.amount,
        result.supported_by_policy,
    )

    return QueryResponse(
        request_id=request_id,
        latency_seconds=round(latency, 2),
        result=result,
    )