from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.models.entities import RagDocument
from backend.utils.config import settings


def build_with_chromadb(records, persist_dir: str):
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("patient_history")

    ids = [r["embedding_id"] for r in records]
    docs = [r["document_text"] for r in records]
    metadatas = [{"patient_db_id": r["patient_db_id"], "scan_id": r["scan_id"]} for r in records]

    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metadatas)


def build_fallback(records, persist_dir: str):
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [r["document_text"] for r in records]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(docs) if docs else None

    payload = {
        "records": records,
        "vectorizer_vocabulary": vectorizer.vocabulary_,
        "matrix_shape": list(matrix.shape) if matrix is not None else [0, 0],
    }
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    Path(persist_dir, "fallback_index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(output_dir: str | None = None):
    out_dir = output_dir or settings.vector_store_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    engine = create_engine(settings.db_url, future=True, connect_args={"check_same_thread": False})
    with Session(engine) as db:
        rows = db.scalars(select(RagDocument)).all()

    records = [
        {
            "id": row.id,
            "patient_db_id": row.patient_db_id,
            "scan_id": row.scan_id,
            "document_text": row.document_text,
            "embedding_id": row.embedding_id,
        }
        for row in rows
    ]

    try:
        build_with_chromadb(records, out_dir)
        print(f"Built ChromaDB index at: {out_dir}")
    except Exception:
        build_fallback(records, out_dir)
        print(f"ChromaDB unavailable; built TF-IDF fallback index at: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vector index from RAG documents")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    main(args.output_dir)
