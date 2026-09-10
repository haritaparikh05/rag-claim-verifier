# RAG Claim Verifier

Given a product claim (e.g. "waterproof to 30 meters," "BPA-free," "12-hour battery life") and a product's official documentation, this system returns a verdict - supported, contradicted, or unverifiable - backed by the exact passage from the source spec sheet it's grounding the answer in.

It's built as a verification task rather than a chatbot: given a claim and evidence, classify whether the evidence actually supports the claim, instead of generating a free-form answer that might hallucinate.

## Status

Early development. See commit history for progress - this README will be filled in with architecture, setup instructions, and sample requests once the core pipeline is working end to end.

## Stack

- Python, FastAPI
- PostgreSQL + pgvector (AWS RDS in production, Docker Compose locally)
- S3 for source spec sheet storage
- Docker, ECS Fargate for deployment
- GitHub Actions for CI/CD
- pytest for testing
