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


def add_rag_document(
    db: Session,
    patient_db_id: int,
    scan_id: int,
    document_text: str,
    document_type: str = "scan_summary",
) -> RagDocument:
    embedding_id = f"patient_{patient_db_id}_scan_{scan_id}_{document_type}"
    doc = RagDocument(
        patient_db_id=patient_db_id,
        scan_id=scan_id,
        document_type=document_type,
        document_text=document_text,
        embedding_id=embedding_id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def _rank_documents(query: str, docs: list[str], top_k: int) -> list[int]:
    if not docs:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_matrix = vectorizer.fit_transform(docs)
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, doc_matrix).flatten()
    ranked_indices = sims.argsort()[::-1][: max(1, top_k)]
    return [int(idx) for idx in ranked_indices]


def _scan_fact(scan: Scan) -> str:
    if scan.tumor_volume is not None:
        metric = f"{scan.tumor_volume:.2f} mm^3"
    else:
        metric = f"{scan.tumor_area:.2f} pixels (2D area approximation)"
    return (
        f"scan_id={scan.id}, scan_date={scan.scan_date}, "
        f"tumor_detected={scan.tumor_detected}, tumor_type={scan.tumor_type or 'No_Tumor'}, "
        f"stage={scan.stage_label or 'N/A'}, risk={scan.risk_category}, "
        f"confidence={scan.confidence_score:.3f}, metric={metric}"
    )


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

    patient_texts = [d.document_text for d in docs]
    ranked_patient_idxs = _rank_documents(query, patient_texts, top_k=max(2, top_k))

    selected_patient_context: list[str] = []
    citations: list[str] = []
    facts: list[str] = []
    for idx in ranked_patient_idxs:
        d = docs[idx]
        scan = db.get(Scan, d.scan_id)
        if scan:
            cite = f"patient_record:scan_id={scan.id}, scan_date={scan.scan_date}, doc_type={d.document_type}"
            facts.append(_scan_fact(scan))
        else:
            cite = f"patient_record:scan_id={d.scan_id}, doc_type={d.document_type}"
        citations.append(cite)
        selected_patient_context.append(d.document_text.strip())

    patient_summary = " ".join(selected_patient_context[:3]).strip()
    if not patient_summary:
        return RagAnswer(
            grounded_answer="No stored scan history is available for this patient.",
            citations=[],
        )

    facts_text = " | ".join(facts[:3]) if facts else "No structured scan-level facts available."
    grounded_answer = (
        f"Grounded historical summary for patient {patient_id} based only on stored patient records: "
        f"{patient_summary} "
        f"Relevant stored scan outputs: {facts_text}."
    )

    return RagAnswer(grounded_answer=grounded_answer, citations=citations)
