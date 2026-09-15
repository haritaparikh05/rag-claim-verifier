import os
import re
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

LANGUAGE_MARKER_RE = re.compile(r"^\s*\[([A-Z]{2})\]\s*$", re.MULTILINE)


def _is_prose_line(line: str) -> bool:
    """A long line containing letters reads as a real sentence fragment.
    isalpha() (unlike islower()) is true for CJK ideographs too, which have
    no case distinction, so this also catches Chinese/Japanese/Korean prose."""
    return len(line) > 25 and any(c.isalpha() for c in line)


def restrict_to_english_section(text: str) -> str:
    """Some manuals repeat the same instructions in multiple languages, tagged
    with markers like [EN], [FR], [DA]. Claims are always in English, so the
    [EN] section is kept in full and other-language sentences are dropped.
    Some manuals also mix in shared, language-neutral reference content
    (icon legends, numeric spec tables) under a non-English marker purely
    because of page layout - since that's short label/number lines rather
    than sentences, it survives a line-level (not whole-segment) prose
    filter applied to the non-English sections."""
    markers = list(LANGUAGE_MARKER_RE.finditer(text))
    if not markers:
        return text

    segments = []
    for i, m in enumerate(markers):
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        segments.append((m.group(1), text[start:end]))

    kept_parts = []
    for lang, seg in segments:
        if lang == "EN":
            kept_parts.append(seg)
        else:
            non_prose_lines = [
                line for line in seg.split("\n") if line.strip() and not _is_prose_line(line.strip())
            ]
            if non_prose_lines:
                kept_parts.append("\n".join(non_prose_lines))

    return "\n".join(kept_parts) if kept_parts else text


def extract_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return restrict_to_english_section(text)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".md" and "\n---\n" in text:
        text = text.split("\n---\n", 1)[1]
    return text


SECTION_HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 ,:&/'\-]{2,60}$")


def _is_section_header(line: str) -> bool:
    """An all-caps short line reads as a section header (e.g. 'TECHNICAL
    SPECIFICATIONS', 'FEATURES:') rather than a sentence."""
    return bool(SECTION_HEADER_RE.match(line))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # Start a fresh chunk at a section header instead of merging it into
        # whatever came before - otherwise a spec table can end up sharing a
        # chunk with unrelated preceding content (e.g. safety warnings),
        # diluting the embedding of the actual spec text.
        starts_new_section = bool(current) and _is_section_header(para)
        fits = len(current) + len(para) + 1 <= chunk_size

        if not starts_new_section and fits:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            overlap_text = "" if starts_new_section else (current[-overlap:] if current else "")
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
