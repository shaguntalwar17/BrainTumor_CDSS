from __future__ import annotations

from pydantic import BaseModel


class APIMessage(BaseModel):
    message: str
