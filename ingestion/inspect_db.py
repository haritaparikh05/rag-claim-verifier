import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "claimverifier"),
        user=os.getenv("DB_USER", "claimverifier"),
        password=os.getenv("DB_PASSWORD", "claimverifier"),
    )


def main():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT product_id, COUNT(*) FROM document_chunks GROUP BY product_id ORDER BY product_id"
    )
    print("Chunks per product:")
    for product_id, count in cur.fetchall():
        print(f"  {product_id}: {count}")

    cur.execute(
        "SELECT product_id, chunk_text FROM document_chunks ORDER BY product_id, chunk_index LIMIT 1"
    )
    row = cur.fetchone()
    if row:
        print("\nSample chunk:")
        print(f"  product_id: {row[0]}")
        print(f"  text: {row[1][:300]}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
