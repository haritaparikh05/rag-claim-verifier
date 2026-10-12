import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.batch import MAX_BATCH_SIZE, MAX_CONCURRENCY, MIN_SECONDS_BETWEEN_STARTS, run_batch
from app.errors import NO_DOCUMENTS_MESSAGE, friendly_error
from app.logging_config import configure_logging
from app.products import PRODUCT_CATALOG, PRODUCT_IDS
from app.results import load_results
from app.retrieval import check_database, count_chunks_by_product
from app.verdict import verify_claim

configure_logging()
logger = logging.getLogger("app")

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_CLAIM_LENGTH = 500

app = FastAPI(title="RAG Claim Verifier")


class VerifyRequest(BaseModel):
    product_id: str = Field(..., examples=["black_diamond_spot_400"])
    claim: str = Field(..., min_length=1, max_length=MAX_CLAIM_LENGTH, examples=["waterproof to 1 meter for 30 minutes"])

    @field_validator("product_id", "claim", mode="before")
    @classmethod
    def strip_whitespace(cls, value):
        return value.strip() if isinstance(value, str) else value


class VerifyResponse(BaseModel):
    verdict: str
    confidence: float
    cited_passage: str
    explanation: str
    latency_ms: int


class BatchRequest(BaseModel):
    product_id: str
    claims: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)

    @field_validator("product_id", mode="before")
    @classmethod
    def strip_product_id(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("claims", mode="before")
    @classmethod
    def strip_claims(cls, value):
        return [c.strip() if isinstance(c, str) else c for c in value] if isinstance(value, list) else value

    @field_validator("claims")
    @classmethod
    def claims_must_be_usable(cls, value):
        if any(not claim or len(claim) > MAX_CLAIM_LENGTH for claim in value):
            raise ValueError("every claim must be non-empty and within the length limit")
        return value


def _validation_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    fields = {str(err["loc"][1]) if len(err["loc"]) > 1 else str(err["loc"][-1]) for err in errors if err.get("loc")}
    if "product_id" in fields:
        return "Please choose a product."
    if "claims" in fields:
        if any(err.get("type") == "too_long" for err in errors):
            return f"A batch can contain at most {MAX_BATCH_SIZE} claims."
        if any(err.get("type") in ("too_short", "missing") for err in errors):
            return "Please add at least one claim."
        return f"Every claim must be filled in and no longer than {MAX_CLAIM_LENGTH} characters."
    if "claim" in fields:
        if any(err.get("type") == "string_too_long" for err in errors):
            return f"That claim is too long (the limit is {MAX_CLAIM_LENGTH} characters)."
        return "Please enter a claim to check."
    return "The request was not valid."


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": _validation_message(exc)})


def _raise_friendly(exc: Exception) -> None:
    """Re-raise a known failure as an HTTP error with a readable message; unexpected errors propagate unchanged."""
    mapped = friendly_error(exc)
    if mapped is None:
        raise exc
    status, message = mapped
    logger.warning("request_failed_known", extra={"fields": {"status": status, "reason": type(exc).__name__}})
    raise HTTPException(status_code=status, detail=message) from exc


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


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/results", include_in_schema=False)
def results_page():
    return FileResponse(STATIC_DIR / "results.html")


@app.get("/api/results")
def api_results():
    """Evaluation scores, computed on each request from the result files committed in data/eval."""
    return load_results()


@app.get("/health")
def health():
    if check_database():
        return {"status": "ok", "database": "ok"}
    logger.error("health_check_failed", extra={"fields": {"database": "unreachable"}})
    return JSONResponse(status_code=503, content={"status": "degraded", "database": "unreachable"})


@app.get("/products")
def products():
    """Every product the system has documentation for, with how many chunks are loaded (0 means not ingested)."""
    try:
        counts = count_chunks_by_product()
    except Exception as exc:
        _raise_friendly(exc)
    return [{**product, "chunks": counts.get(product["product_id"], 0)} for product in PRODUCT_CATALOG]


@app.post("/verify", response_model=VerifyResponse)
def verify(body: VerifyRequest, http_request: Request):
    if body.product_id not in PRODUCT_IDS:
        raise HTTPException(status_code=404, detail=f"Unknown product '{body.product_id}'. Please choose one from the list.")

    start = time.perf_counter()
    try:
        result = verify_claim(body.product_id, body.claim)
    except Exception as exc:
        _raise_friendly(exc)
    latency_ms = round((time.perf_counter() - start) * 1000)

    if not result.get("retrieved_chunks"):
        raise HTTPException(status_code=503, detail=NO_DOCUMENTS_MESSAGE)

    logger.info(
        "verify",
        extra={
            "fields": {
                "request_id": http_request.state.request_id,
                "product_id": body.product_id,
                "claim": body.claim[:200],
                "verdict": result["verdict"],
                "confidence": result["confidence"],
                "retrieved_chunks": len(result["retrieved_chunks"]),
                "latency_ms": latency_ms,
            }
        },
    )
    return VerifyResponse(
        verdict=result["verdict"],
        confidence=result["confidence"],
        cited_passage=result["cited_passage"],
        explanation=result["explanation"],
        latency_ms=latency_ms,
    )


def _ndjson(payload: dict) -> str:
    return json.dumps(payload) + "\n"


@app.post("/verify/batch")
def verify_batch(body: BatchRequest, http_request: Request):
    """Check up to MAX_BATCH_SIZE claims for one product.

    The response streams one JSON object per line as each claim finishes: a "start" line, a "result" line per claim
    (in completion order, each carrying its input `index`; failed claims carry `error` and `status`), then "done".
    """
    if body.product_id not in PRODUCT_IDS:
        raise HTTPException(status_code=404, detail=f"Unknown product '{body.product_id}'. Please choose one from the list.")

    request_id = http_request.state.request_id

    def stream():
        started = time.perf_counter()
        succeeded = failed = 0
        yield _ndjson({"type": "start", "total": len(body.claims), "max_concurrency": MAX_CONCURRENCY})
        for item in run_batch(
            body.product_id,
            body.claims,
            verify_claim,
            max_workers=MAX_CONCURRENCY,
            min_interval=MIN_SECONDS_BETWEEN_STARTS,
        ):
            if item["error"]:
                failed += 1
            else:
                succeeded += 1
            yield _ndjson({"type": "result", **item})
        duration_ms = round((time.perf_counter() - started) * 1000)
        logger.info(
            "verify_batch",
            extra={
                "fields": {
                    "request_id": request_id,
                    "product_id": body.product_id,
                    "total": len(body.claims),
                    "succeeded": succeeded,
                    "failed": failed,
                    "duration_ms": duration_ms,
                }
            },
        )
        yield _ndjson({"type": "done", "succeeded": succeeded, "failed": failed, "duration_ms": duration_ms})

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# Mounted last so it never shadows an API route.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
