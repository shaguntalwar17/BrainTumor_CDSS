from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models.entities import Patient, RagDocument, Scan


@dataclass
class RagAnswer:
    grounded_answer: str
    citations: list[str]


def add_rag_document(db: Session, patient_db_id: int, scan_id: int, document_text: str) -> RagDocument:
    embedding_id = f"patient_{patient_db_id}_scan_{scan_id}"
    doc = RagDocument(
        patient_db_id=patient_db_id,
        scan_id=scan_id,
        document_type="scan_summary",
        document_text=document_text,
        embedding_id=embedding_id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def query_patient_history(db: Session, patient_id: str, query: str, top_k: int = 5) -> RagAnswer:
    patient = db.scalar(select(Patient).where(Patient.patient_id == patient_id))
    if not patient:
        return RagAnswer(
            grounded_answer="No stored scan history is available for this patient.",
            citations=[],
        )

    docs = db.scalars(select(RagDocument).where(RagDocument.patient_db_id == patient.id)).all()
    if not docs:
        return RagAnswer(
            grounded_answer="No stored scan history is available for this patient.",
            citations=[],
        )

    texts = [d.document_text for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_matrix = vectorizer.fit_transform(texts)
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, doc_matrix).flatten()

    ranked_indices = sims.argsort()[::-1][: max(1, top_k)]

    selected = []
    citations = []
    for idx in ranked_indices:
        d = docs[int(idx)]
        scan = db.get(Scan, d.scan_id)
        if scan:
            cite = f"scan_id={scan.id}, scan_date={scan.scan_date}"
        else:
            cite = f"scan_id={d.scan_id}"
        citations.append(cite)
        selected.append(d.document_text)

    grounded_answer = (
        f"Grounded historical summary for patient {patient_id}: "
        + " ".join(selected[:3])
        + " This response is grounded strictly in stored patient records."
    )

    return RagAnswer(grounded_answer=grounded_answer, citations=citations)
