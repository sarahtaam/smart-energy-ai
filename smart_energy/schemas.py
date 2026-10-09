"""Validation of the telemetry payload sent by ESP32 nodes."""
import time
from typing import Optional

from pydantic import BaseModel, Field, field_validator

FIELDS = ["voltage", "current", "power", "energy", "frequency", "pf", "temperature", "humidity"]


class Telemetry(BaseModel):
    device_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    ts: Optional[int] = None  # unix seconds (UTC); filled with receive time if missing
    voltage: Optional[float] = Field(default=None, ge=0, le=400)
    current: Optional[float] = Field(default=None, ge=0, le=100)
    power: Optional[float] = Field(default=None, ge=0, le=25000)
    energy: Optional[float] = Field(default=None, ge=0)
    frequency: Optional[float] = Field(default=None, ge=0, le=70)
    pf: Optional[float] = Field(default=None, ge=0, le=1)
    temperature: Optional[float] = Field(default=None, ge=-40, le=85)
    humidity: Optional[float] = Field(default=None, ge=0, le=100)

    @field_validator("ts")
    @classmethod
    def _ts_sane(cls, v):
        # Reject timestamps from an unsynced RTC (1970) or far in the future
        if v is not None and not (1_600_000_000 <= v <= time.time() + 3600):
            return None
        return v

    def with_ts(self) -> "Telemetry":
        if self.ts is None:
            return self.model_copy(update={"ts": int(time.time())})
        return self
