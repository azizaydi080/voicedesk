"""Energy-based voice activity detection and endpointing with an adaptive noise floor.

Deliberately simple and dependency-free: it runs per 20 ms frame in microseconds, and the
pipeline only needs two decisions from it, "the caller started talking" (for barge-in)
and "the caller stopped talking" (endpointing). The browser's echo cancellation keeps
the agent's own voice out of the microphone signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from voicedesk.providers.base import SAMPLE_RATE

FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2


class VadEvent(Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


def frame_dbfs(frame: bytes) -> float:
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
    return 20 * math.log10(max(rms, 1.0) / 32768.0)


@dataclass
class EnergyVAD:
    threshold_db: float = 12.0
    start_ms: int = 60
    end_silence_ms: int = 550
    min_abs_db: float = -50.0
    floor_db: float = -65.0
    calibration_ms: int = 200
    max_utterance_ms: int = 20_000
    speaking: bool = False
    _elapsed_ms: int = 0
    _utterance_ms: int = 0
    _voiced_ms: int = 0
    _silence_ms: int = 0
    _pending: bytearray = field(default_factory=bytearray)

    def process(self, pcm16: bytes) -> list[tuple[bytes, VadEvent | None]]:
        """Split ``pcm16`` into 20 ms frames, returning each frame with the event it triggered."""
        self._pending.extend(pcm16)
        out: list[tuple[bytes, VadEvent | None]] = []
        while len(self._pending) >= FRAME_BYTES:
            frame = bytes(self._pending[:FRAME_BYTES])
            del self._pending[:FRAME_BYTES]
            out.append((frame, self._step(frame)))
        return out

    def _step(self, frame: bytes) -> VadEvent | None:
        db = frame_dbfs(frame)
        self._elapsed_ms += FRAME_MS
        if self._elapsed_ms <= self.calibration_ms:
            # Learn the room's background level from the first frames after the mic opens.
            self.floor_db = db if self._elapsed_ms == FRAME_MS else 0.7 * self.floor_db + 0.3 * db
            return None
        voiced = db > self.floor_db + self.threshold_db and db > self.min_abs_db
        # Minimum tracking: follow drops quickly and rises slowly (~2 s time constant), so a
        # steady fan or hum is absorbed into the floor while word-level dips keep speech low.
        rate = 0.2 if db < self.floor_db else 0.01
        self.floor_db = max(-90.0, self.floor_db + rate * (db - self.floor_db))
        if not self.speaking:
            self._voiced_ms = (
                self._voiced_ms + FRAME_MS if voiced else max(0, self._voiced_ms - FRAME_MS)
            )
            if self._voiced_ms >= self.start_ms:
                self.speaking, self._silence_ms, self._utterance_ms = True, 0, 0
                return VadEvent.SPEECH_START
            return None
        self._utterance_ms += FRAME_MS
        self._silence_ms = 0 if voiced else self._silence_ms + FRAME_MS
        if self._silence_ms >= self.end_silence_ms or self._utterance_ms >= self.max_utterance_ms:
            self.speaking, self._voiced_ms = False, 0
            return VadEvent.SPEECH_END
        return None
