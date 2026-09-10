import os
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def extract_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".md" and "\n---\n" in text:
        text = text.split("\n---\n", 1)[1]
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            overlap_text = current[-overlap:] if current else ""
            current = f"{overlap_text}\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


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


def main():
    model = SentenceTransformer(EMBEDDING_MODEL)
    conn = get_db_connection()
    cur = conn.cursor()

    product_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    total_chunks = 0

    for product_dir in product_dirs:
        product_id = product_dir.name
        cur.execute("DELETE FROM document_chunks WHERE product_id = %s", (product_id,))

        files = sorted(f for f in product_dir.iterdir() if f.is_file())
        product_chunks = []
        for file_path in files:
            text = extract_text(file_path)
            for i, chunk in enumerate(chunk_text(text)):
                product_chunks.append((product_id, file_path.name, i, chunk))

        if not product_chunks:
            print(f"[{product_id}] no chunks extracted, skipping")
            continue

        texts = [c[3] for c in product_chunks]
        embeddings = model.encode(texts, show_progress_bar=False)

        rows = [
            (pid, src, idx, chunk, emb)
            for (pid, src, idx, chunk), emb in zip(product_chunks, embeddings)
        ]

        execute_values(
            cur,
            "INSERT INTO document_chunks (product_id, source_file, chunk_index, chunk_text, embedding) VALUES %s",
            rows,
        )
        conn.commit()
        total_chunks += len(rows)
        print(f"[{product_id}] inserted {len(rows)} chunks from {len(files)} file(s)")

    cur.close()
    conn.close()
    print(f"Done. {total_chunks} chunks across {len(product_dirs)} products.")


if __name__ == "__main__":
    main()
