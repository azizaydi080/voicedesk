"""HTTP + WebSocket API.

WebSocket protocol (``/ws``):

client -> server
  binary frames          PCM16 mono 16 kHz audio from the microphone (any frame size)
  {"type": "text", "text": "..."}                    typed turn (skips STT)
  {"type": "playback_started", "turn_id", "client_ttfa_ms"}
  {"type": "playback_ended"} | {"type": "interrupt"}

server -> client
  {"type": "ready", "session_id", "providers"}
  {"type": "vad", "event": "speech_start" | "speech_end"}
  {"type": "transcript", "final": bool, "text", "language"?, ...}
  {"type": "state", "state": "thinking" | "speaking"}
  {"type": "agent_text", "delta"} | {"type": "tool_call", ...} | {"type": "tool_result", ...}
  {"type": "audio", "seq", "format", "text", "bytes"} followed by one binary frame
  {"type": "interrupt", "reason"} | {"type": "turn_end", "metrics"} | {"type": "error"}
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from voicedesk.config import Settings, get_settings
from voicedesk.metrics.store import MetricsStore, summarize
from voicedesk.pipeline.session import Sender, VoiceSession
from voicedesk.providers.registry import Providers, build_providers

log = logging.getLogger("voicedesk")


def create_app(settings: Settings | None = None, providers: Providers | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.providers = providers or await asyncio.to_thread(build_providers, settings)
        app.state.metrics = MetricsStore(settings.metrics_path)
        p: Providers = app.state.providers
        await p.stt.warmup()
        # Prime the LLM's prompt cache with the real system prompt + tool schemas.
        primer = VoiceSession(p, settings, Sender(_noop_json, _noop_audio))
        await primer.agent.warmup()
        log.info("ready: stt=%s llm=%s tts=%s", p.stt.name, p.llm.name, p.tts.name)
        yield

    app = FastAPI(title="VoiceDesk voice agent", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        p: Providers = app.state.providers
        return {"ok": True, "providers": {"stt": p.stt.name, "llm": p.llm.name, "tts": p.tts.name}}

    @app.get("/api/metrics")
    async def metrics(limit: int = 200) -> dict[str, Any]:
        turns = app.state.metrics.recent(min(limit, 2000))
        return {"summary": summarize(turns), "turns": turns}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        lock = asyncio.Lock()  # an audio announcement and its binary frame must stay adjacent

        async def send_json(msg: dict[str, Any]) -> None:
            async with lock:
                await ws.send_text(json.dumps(msg, ensure_ascii=False))

        async def send_audio(meta: dict[str, Any], data: bytes) -> None:
            async with lock:
                await ws.send_text(json.dumps(meta, ensure_ascii=False))
                await ws.send_bytes(data)

        p: Providers = app.state.providers
        session = VoiceSession(p, settings, Sender(send_json, send_audio), app.state.metrics)
        await send_json(
            {
                "type": "ready",
                "session_id": session.session_id,
                "providers": {"stt": p.stt.name, "llm": p.llm.name, "tts": p.tts.name},
                "today": session.store.today.isoformat(),
            }
        )
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    await session.handle_audio(msg["bytes"])
                elif msg.get("text"):
                    data = json.loads(msg["text"])
                    if data.get("type") == "text" and str(data.get("text", "")).strip():
                        await session.handle_text(str(data["text"]))
                    else:
                        await session.handle_control(data)
        except WebSocketDisconnect:
            pass
        finally:
            await session.close()

    return app


async def _noop_json(_: dict[str, Any]) -> None:
    return None


async def _noop_audio(_: dict[str, Any], __: bytes) -> None:
    return None
