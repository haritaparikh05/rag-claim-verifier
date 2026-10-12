FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model now so the first request never waits on the network; the offline flag
# (set after the download) keeps the running container from contacting the Hugging Face Hub at all.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY app/ app/
COPY ingestion/ ingestion/
COPY data/raw/ data/raw/
COPY data/eval/ data/eval/
COPY demo/ demo/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
