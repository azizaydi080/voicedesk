"""Per-turn latency records, persisted to SQLite so the dashboard survives restarts."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STAGES = (
    "stt_ms",
    "llm_ttft_ms",
    "tool_ms",
    "tts_ttfb_ms",
    "tts_first_chunk_ms",
    "ttfa_ms",
    "client_ttfa_ms",
    "turn_ms",
)


@dataclass
class TurnMetrics:
    turn_id: str
    session_id: str
    started_at: float
    input_mode: str  # "voice" | "text"
    providers: dict[str, str]
    endpoint_silence_ms: int = 0
    audio_seconds: float | None = None
    transcript: str = ""
    language: str | None = None
    language_method: str | None = None
    stt_language: str | None = None
    reply: str = ""
    tools: list[str] = field(default_factory=list)
    llm_rounds: int = 0
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    first_chunk_chars: int | None = None
    # Latencies in ms. ttfa = end of caller speech (endpoint fired) -> first audio byte sent.
    stt_ms: float | None = None
    llm_ttft_ms: float | None = None
    llm_total_ms: float | None = None
    tool_ms: float | None = None
    tts_ttfb_ms: float | None = None
    tts_first_chunk_ms: float | None = None  # full synthesis of the first chunk
    ttfa_ms: float | None = None
    client_ttfa_ms: float | None = None
    turn_ms: float | None = None
    interrupted: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def summarize(turns: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"turns": len(turns)}
    for stage in STAGES:
        vals = [float(t[stage]) for t in turns if t.get(stage) is not None]
        out[stage] = {
            "n": len(vals),
            "p50": percentile(vals, 50),
            "p90": percentile(vals, 90),
            "p95": percentile(vals, 95),
        }
    out["interrupted"] = sum(1 for t in turns if t.get("interrupted"))
    return out


class MetricsStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS turns (turn_id TEXT PRIMARY KEY, session_id TEXT, "
            "started_at REAL, data TEXT)"
        )
        self._db.commit()

    def save(self, m: TurnMetrics) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO turns VALUES (?, ?, ?, ?)",
                (
                    m.turn_id,
                    m.session_id,
                    m.started_at,
                    json.dumps(m.as_dict(), ensure_ascii=False),
                ),
            )
            self._db.commit()

    def patch(self, turn_id: str, **fields: Any) -> None:
        with self._lock:
            row = self._db.execute(
                "SELECT data FROM turns WHERE turn_id = ?", (turn_id,)
            ).fetchone()
            if row is None:
                return
            data = json.loads(row[0]) | fields
            self._db.execute(
                "UPDATE turns SET data = ? WHERE turn_id = ?",
                (json.dumps(data, ensure_ascii=False), turn_id),
            )
            self._db.commit()

    def recent(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT data FROM turns ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]
