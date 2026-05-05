from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.models.entities import Patient


@dataclass
class PatientResolution:
    patient: Patient
    matched_existing: bool
    generated_new_id: bool
    match_strategy: str
    match_score: float | None = None


def _clean(value: str | None) -> str:
    if not value:
        return ""
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _profile_signature(name: str, age: int, gender: str, contact: str | None) -> str:
    return f"name={_clean(name)} | age={int(age)} | gender={_clean(gender)} | contact={_clean(contact)}"


def _name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, _clean(a), _clean(b)).ratio())


def _tfidf_similarity(query: str, candidates: list[str]) -> list[float]:
    if not candidates:
        return []
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform([query, *candidates])
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return [float(v) for v in sims]


def _next_patient_id(db: Session) -> str:
    prefix = f"PAT-{datetime.utcnow():%Y}-"
    rows = db.scalars(select(Patient.patient_id).where(Patient.patient_id.like(f"{prefix}%"))).all()
    max_seq = 0
    for value in rows:
        if not value:
            continue
        tail = value.replace(prefix, "", 1)
        if tail.isdigit():
            max_seq = max(max_seq, int(tail))
    return f"{prefix}{max_seq + 1:04d}"


def resolve_or_create_patient(
    db: Session,
    provided_patient_id: str | None,
    patient_name: str,
    age: int,
    gender: str,
    contact: str | None,
) -> PatientResolution:
    signature = _profile_signature(patient_name, age, gender, contact)
    provided_id = _clean(provided_patient_id)

    if provided_id:
        patient = db.scalar(
            select(Patient).where(or_(Patient.patient_id == provided_patient_id, Patient.patient_code == provided_patient_id))
        )
        if patient:
            patient.name = patient_name
            patient.age = age
            patient.gender = gender
            patient.contact = contact
            patient.profile_signature = signature
            db.commit()
            db.refresh(patient)
            return PatientResolution(
                patient=patient,
                matched_existing=True,
                generated_new_id=False,
                match_strategy="exact_patient_id",
                match_score=1.0,
            )

        new_patient = Patient(
            patient_id=provided_patient_id.strip(),
            patient_code=provided_patient_id.strip(),
            name=patient_name,
            age=age,
            gender=gender,
            contact=contact,
            profile_signature=signature,
        )
        db.add(new_patient)
        db.commit()
        db.refresh(new_patient)
        return PatientResolution(
            patient=new_patient,
            matched_existing=False,
            generated_new_id=False,
            match_strategy="manual_patient_id_new_record",
        )

    candidates = db.scalars(select(Patient).order_by(Patient.created_at.desc()).limit(2000)).all()
    if candidates:
        candidate_signatures = [
            c.profile_signature or _profile_signature(c.name, c.age, c.gender, c.contact)
            for c in candidates
        ]
        sims = _tfidf_similarity(signature, candidate_signatures)
        best_patient: Patient | None = None
        best_score = -1.0
        for idx, patient in enumerate(candidates):
            name_score = _name_similarity(patient_name, patient.name)
            profile_score = sims[idx] if idx < len(sims) else 0.0
            age_delta = abs(int(age) - int(patient.age))
            age_score = max(0.0, 1.0 - (age_delta / 20.0))
            gender_score = 1.0 if _clean(gender) == _clean(patient.gender) else 0.2
            if contact and patient.contact and _clean(contact) == _clean(patient.contact):
                contact_score = 1.0
            elif contact and patient.contact:
                contact_score = 0.0
            else:
                contact_score = 0.3

            score = (
                0.45 * name_score
                + 0.25 * profile_score
                + 0.15 * age_score
                + 0.1 * gender_score
                + 0.05 * contact_score
            )
            if score > best_score:
                best_score = score
                best_patient = patient

        threshold = 0.86 if contact else 0.9
        if best_patient and best_score >= threshold:
            best_patient.name = patient_name
            best_patient.age = age
            best_patient.gender = gender
            best_patient.contact = contact
            best_patient.profile_signature = signature
            db.commit()
            db.refresh(best_patient)
            return PatientResolution(
                patient=best_patient,
                matched_existing=True,
                generated_new_id=False,
                match_strategy="profile_similarity_match",
                match_score=float(best_score),
            )

    new_id = _next_patient_id(db)
    created = Patient(
        patient_id=new_id,
        patient_code=new_id,
        name=patient_name,
        age=age,
        gender=gender,
        contact=contact,
        profile_signature=signature,
    )
    db.add(created)
    db.commit()
    db.refresh(created)
    return PatientResolution(
        patient=created,
        matched_existing=False,
        generated_new_id=True,
        match_strategy="auto_generated_patient_id",
    )
