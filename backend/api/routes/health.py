from fastapi import APIRouter

from backend.utils.disclaimer import ATTRIBUTION, STAGE_NOTE, UI_DISCLAIMER


router = APIRouter(tags=["health"])


@router.get("/")
def root():
    return {
        "service": "Brain MRI Tumor AI Platform API",
        "status": "ok",
        "disclaimer": UI_DISCLAIMER,
        "stage_note": STAGE_NOTE,
        "attribution": ATTRIBUTION,
    }


@router.get("/health")
@router.get("/api/health")
def health():
    return {"status": "healthy"}

