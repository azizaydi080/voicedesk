"""``voicedesk-bench``: end-to-end latency benchmark against a running server.

Plays each scenario's audio fixtures into ``/ws`` in real time (20 ms frames, like a
microphone), then keeps sending silence until the agent's turn ends. Records the server's
per-stage metrics and, independently, the client-side time from the end of the caller's
speech to the first audio byte received, which includes the endpointing window.

    voicedesk-bench --url ws://localhost:8000/ws --repeat 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
from pathlib import Path
from typing import Any

from voicedesk.evals.cli import EVALS, _git_sha
from voicedesk.evals.runner import fixture_path, read_wav
from voicedesk.evals.scoring import load_suite
from voicedesk.metrics.store import STAGES, percentile

FRAME = 640  # 20 ms of PCM16 @ 16 kHz
SILENCE = bytes(FRAME)
TAIL_PAD_S = 0.3  # fixtures end with 300 ms of silence after the last word


async def _turn(ws: Any, pcm: bytes) -> dict[str, Any]:
    """Stream one utterance; return server metrics plus client-side timings."""
    t_speech_end: float | None = None
    t_first_audio: float | None = None
    metrics: dict[str, Any] | None = None
    done = asyncio.Event()

    async def reader() -> None:
        nonlocal t_first_audio, metrics
        async for raw in ws:
            if isinstance(raw, bytes):
                if t_first_audio is None:
                    t_first_audio = time.perf_counter()
                continue
            msg = json.loads(raw)
            if msg["type"] == "turn_end":
                metrics = msg.get("metrics") or {"empty": True}
                done.set()
                return

    task = asyncio.create_task(reader())
    start = time.perf_counter()
    for i in range(0, len(pcm), FRAME):
        await ws.send(pcm[i : i + FRAME].ljust(FRAME, b"\0"))
        # Pace against the wall clock so drift does not accumulate.
        await asyncio.sleep(max(0.0, start + (i + FRAME) / 32_000 - time.perf_counter()))
    t_speech_end = start + len(pcm) / 32_000 - TAIL_PAD_S
    while not done.is_set():  # the "microphone" keeps sending silence
        await ws.send(SILENCE)
        await asyncio.sleep(0.02)
    await task
    await ws.send(json.dumps({"type": "playback_ended"}))
    assert metrics is not None
    if t_first_audio is not None:
        metrics["client_speech_end_to_audio_ms"] = round((t_first_audio - t_speech_end) * 1000, 1)
    return metrics


async def run(url: str, repeat: int, only: set[str] | None) -> list[dict[str, Any]]:
    import websockets

    suite = load_suite(EVALS / "scenarios.yaml")
    rows: list[dict[str, Any]] = []
    for rep in range(repeat):
        for sc in suite.scenarios:
            if only and sc.id not in only:
                continue
            async with websockets.connect(url, max_size=8 * 1024 * 1024) as ws:
                json.loads(await ws.recv())  # ready
                for i, _turn_spec in enumerate(sc.turns):
                    m = await _turn(ws, read_wav(fixture_path(EVALS / "audio", sc.id, i)))
                    m |= {"scenario": sc.id, "turn": i, "repeat": rep}
                    rows.append(m)
                    print(
                        f"  {sc.id}#{i}: ttfa {m.get('ttfa_ms')} ms "
                        f"(stt {m.get('stt_ms')}, llm {m.get('llm_ttft_ms')}, "
                        f"tts {m.get('tts_first_chunk_ms')}) "
                        f"client {m.get('client_speech_end_to_audio_ms')} ms",
                        flush=True,
                    )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [*STAGES, "client_speech_end_to_audio_ms"]
    out: dict[str, Any] = {"turns": len(rows)}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        out[k] = {"n": len(vals), "p50": percentile(vals, 50), "p90": percentile(vals, 90)}
    by_lang: dict[str, list[float]] = {}
    for r in rows:
        if r.get("ttfa_ms") is not None:
            by_lang.setdefault(str(r.get("language")), []).append(float(r["ttfa_ms"]))
    out["ttfa_p50_by_language"] = {k: percentile(v, 50) for k, v in by_lang.items()}
    cached = [r["cached_tokens"] / r["prompt_tokens"] for r in rows
              if r.get("prompt_tokens") and r.get("cached_tokens") is not None]
    out["prompt_cache_hit_ratio_p50"] = percentile(cached, 50)
    return out


def main() -> None:
    p = argparse.ArgumentParser(prog="voicedesk-bench")
    p.add_argument("--url", default="ws://localhost:8000/ws")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--only", help="comma-separated scenario ids")
    p.add_argument("--tag", default="local")
    p.add_argument("--out", type=Path, default=EVALS.parent / "docs" / "benchmarks")
    args = p.parse_args()
    rows = asyncio.run(run(args.url, args.repeat, set(args.only.split(",")) if args.only else None))
    providers = next((r["providers"] for r in rows if r.get("providers")), {})
    report = {
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": _git_sha(),
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}".strip(),
        "providers": providers,
        "summary": summarize_rows(rows),
        "turns": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"bench-{args.tag}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
