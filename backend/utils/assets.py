from __future__ import annotations

import json
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


def volume_manifest_to_urls(manifest_path: str | None) -> tuple[list[str], int | None]:
    if not manifest_path:
        return [], None

    mpath = Path(manifest_path)
    if not mpath.exists():
        return [], None

    try:
        payload = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception:
        return [], None

    selected_slice_index = payload.get("selected_slice_index")
    slice_paths = payload.get("slice_paths", [])
    urls = [to_storage_url(str(p)) for p in slice_paths]
    clean_urls = [u for u in urls if u]
    return clean_urls, selected_slice_index
