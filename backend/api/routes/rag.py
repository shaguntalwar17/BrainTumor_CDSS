from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.schemas.rag import RagQueryRequest, RagQueryResponse
from backend.services.rag_service import query_patient_history


router = APIRouter(tags=["rag"])


@router.post("/rag-query", response_model=RagQueryResponse)
@router.post("/api/rag/query", response_model=RagQueryResponse)
def rag_query(payload: RagQueryRequest, db: Session = Depends(get_db)):
    result = query_patient_history(db, patient_id=payload.patient_id, query=payload.query, top_k=payload.top_k)
    return RagQueryResponse(
        patient_id=payload.patient_id,
        query=payload.query,
        grounded_answer=result.grounded_answer,
        citations=result.citations,
    )

