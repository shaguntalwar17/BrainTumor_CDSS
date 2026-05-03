from __future__ import annotations

from pydantic import BaseModel


class RagQueryRequest(BaseModel):
    patient_id: str
    query: str
    top_k: int = 5


class RagQueryResponse(BaseModel):
    patient_id: str
    query: str
    grounded_answer: str
    citations: list[str]
