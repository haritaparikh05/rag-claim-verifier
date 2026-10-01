import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.logging_config import configure_logging
from app.retrieval import check_database
from app.verdict import verify_claim

configure_logging()
logger = logging.getLogger("app")

app = FastAPI(title="RAG Claim Verifier")


class VerifyRequest(BaseModel):
    product_id: str = Field(..., examples=["black_diamond_spot_400"])
    claim: str = Field(..., examples=["waterproof to 1 meter for 30 minutes"])


class VerifyResponse(BaseModel):
    verdict: str
    confidence: float
    cited_passage: str
    explanation: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()
    fields = {"request_id": request_id, "method": request.method, "path": request.url.path}

    try:
        response = await call_next(request)
    except Exception:
        fields["duration_ms"] = round((time.perf_counter() - start) * 1000)
        logger.exception("request_failed", extra={"fields": fields})
        raise

    response.headers["x-request-id"] = request_id
    fields["status"] = response.status_code
    fields["duration_ms"] = round((time.perf_counter() - start) * 1000)
    # Docker and the deploy job poll /health constantly; only log it when it fails.
    if request.url.path != "/health" or response.status_code != 200:
        logger.info("request", extra={"fields": fields})
    return response


@app.get("/health")
def health():
    if check_database():
        return {"status": "ok", "database": "ok"}
    logger.error("health_check_failed", extra={"fields": {"database": "unreachable"}})
    return JSONResponse(status_code=503, content={"status": "degraded", "database": "unreachable"})


@app.post("/verify", response_model=VerifyResponse)
def verify(body: VerifyRequest, http_request: Request):
    start = time.perf_counter()
    result = verify_claim(body.product_id, body.claim)
    logger.info(
        "verify",
        extra={
            "fields": {
                "request_id": http_request.state.request_id,
                "product_id": body.product_id,
                "claim": body.claim[:200],
                "verdict": result["verdict"],
                "confidence": result["confidence"],
                "retrieved_chunks": len(result.get("retrieved_chunks", [])),
                "latency_ms": round((time.perf_counter() - start) * 1000),
            }
        },
    )
    return VerifyResponse(
        verdict=result["verdict"],
        confidence=result["confidence"],
        cited_passage=result["cited_passage"],
        explanation=result["explanation"],
    )
