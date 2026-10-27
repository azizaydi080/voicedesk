from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from voicedesk.config import Settings
from voicedesk.providers.base import SAMPLE_RATE

TODAY = date(2026, 10, 27)  # a Tuesday


def tone(ms: int, freq: float = 220.0, amp: float = 0.3) -> bytes:
    t = np.arange(int(SAMPLE_RATE * ms / 1000)) / SAMPLE_RATE
    return (np.sin(2 * math.pi * freq * t) * amp * 32767).astype(np.int16).tobytes()


def silence(ms: int) -> bytes:
    return bytes(int(SAMPLE_RATE * ms / 1000) * 2)


@pytest.fixture
def settings(tmp_path: object) -> Settings:
    return Settings(
        stt="fake",
        llm="scripted",
        tts="fake",
        clinic_today=TODAY,
        metrics_path=tmp_path / "metrics.sqlite",  # type: ignore[operator]
        _env_file=None,  # type: ignore[call-arg]
    )
