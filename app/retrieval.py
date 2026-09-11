import argparse
import os
import sys

import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "claimverifier"),
        user=os.getenv("DB_USER", "claimverifier"),
        password=os.getenv("DB_PASSWORD", "claimverifier"),
    )
    register_vector(conn)
    return conn


def retrieve_chunks(product_id: str, claim: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return the top_k chunks for `product_id` most similar to `claim`, ordered by cosine distance (lower = more similar)."""
    model = get_model()
    claim_embedding = model.encode(claim)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT source_file, chunk_index, chunk_text, embedding <=> %s AS distance
        FROM document_chunks
        WHERE product_id = %s
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (claim_embedding, product_id, claim_embedding, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "source_file": source_file,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "distance": float(distance),
        }
        for source_file, chunk_index, chunk_text, distance in rows
    ]


def main():
    # Windows terminals often default to a limited codepage (cp1252) that
    # can't encode every character extracted from PDFs (accents, CJK, etc.)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Test retrieval for a product + claim")
    parser.add_argument("product_id", help="e.g. black_diamond_spot_400")
    parser.add_argument("claim", help="e.g. 'waterproof to 1 meter for 30 minutes'")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    results = retrieve_chunks(args.product_id, args.claim, args.top_k)
    if not results:
        print(f"No chunks found for product_id={args.product_id!r}. Check it exists in the DB.")
        return

    print(f"Claim: {args.claim!r}\nProduct: {args.product_id}\n")
    for i, r in enumerate(results, start=1):
        print(f"[{i}] distance={r['distance']:.4f}  ({r['source_file']}, chunk {r['chunk_index']})")
        print(f"    {r['chunk_text'][:300]}")
        print()


if __name__ == "__main__":
    main()
