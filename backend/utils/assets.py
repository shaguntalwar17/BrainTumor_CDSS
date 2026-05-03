from __future__ import annotations

from pathlib import Path

from backend.utils.config import settings


def to_storage_url(path: str | None) -> str | None:
    if not path:
        return None

    storage_root = Path(settings.storage_root).resolve()
    target = Path(path).resolve()

    try:
        rel = target.relative_to(storage_root)
    except ValueError:
        # Best-effort fallback for relative paths already rooted under backend/storage.
        raw = Path(path).as_posix()
        marker = "backend/storage/"
        if marker in raw:
            rel = Path(raw.split(marker, 1)[1])
        else:
            return None

    return f"/storage/{rel.as_posix()}"

