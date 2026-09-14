from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.verdict import verify_claim

app = FastAPI(title="RAG Claim Verifier")


class VerifyRequest(BaseModel):
    product_id: str = Field(..., examples=["black_diamond_spot_400"])
    claim: str = Field(..., examples=["waterproof to 1 meter for 30 minutes"])


class VerifyResponse(BaseModel):
    verdict: str
    confidence: float
    cited_passage: str
    explanation: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest):
    result = verify_claim(request.product_id, request.claim)
    return VerifyResponse(
        verdict=result["verdict"],
        confidence=result["confidence"],
        cited_passage=result["cited_passage"],
        explanation=result["explanation"],
    )
