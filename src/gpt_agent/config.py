from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8888, ge=1, le=65535)
    model: str = "gpt-6.1-sol"
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_concurrency: int = Field(default=4, ge=1)
    max_retries: int = Field(default=2, ge=0)
    request_timeout_seconds: float = Field(default=90.0, gt=0)
