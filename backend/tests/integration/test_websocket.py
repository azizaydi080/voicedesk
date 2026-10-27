"""Drives the real FastAPI app over a WebSocket with fake providers."""

import json

import pytest
from fastapi.testclient import TestClient

from voicedesk.api.app import create_app
from voicedesk.config import Settings
from voicedesk.providers.base import Transcript
from voicedesk.providers.fakes import FakeSTT, FakeTTS, ScriptedLLM
from voicedesk.providers.registry import Providers
from tests.conftest import silence, tone

pytestmark = pytest.mark.integration


def receive_until_turn_end(ws) -> tuple[list[dict], list[bytes]]:
    events, audio = [], []
    while True:
        msg = ws.receive()
        if msg.get("bytes") is not None:
            audio.append(msg["bytes"])
            continue
        event = json.loads(msg["text"])
        events.append(event)
        if event["type"] == "turn_end":
            return events, audio


def test_text_and_voice_turns_over_websocket(settings: Settings) -> None:
    providers = Providers(
        FakeSTT([Transcript("Are you open on Saturday? What are the hours?")]),
        ScriptedLLM(),
        FakeTTS(),
    )
    with (
        TestClient(create_app(settings, providers)) as client,
        client.websocket_connect("/ws") as ws,
    ):
        ready = json.loads(ws.receive_text())
        assert ready["type"] == "ready" and ready["providers"]["llm"] == "scripted"

        ws.send_text(json.dumps({"type": "text", "text": "Where is my booking 1001?"}))
        events, audio = receive_until_turn_end(ws)
        audio_meta = [e for e in events if e["type"] == "audio"]
        assert len(audio_meta) == len(audio) >= 1
        assert [e["type"] for e in events if e["type"].startswith("tool")] == [
            "tool_call",
            "tool_result",
        ]
        turn_id = events[-1]["turn_id"]
        ws.send_text(
            json.dumps({"type": "playback_started", "turn_id": turn_id, "client_ttfa_ms": 42.0})
        )

        ws.send_bytes(silence(300) + tone(800) + silence(800))
        events, _ = receive_until_turn_end(ws)
        tool = next(e for e in events if e["type"] == "tool_call")
        assert tool["name"] == "clinic_info"

        body = client.get("/api/metrics").json()
        assert body["summary"]["turns"] == 2
        assert {t["input_mode"] for t in body["turns"]} == {"text", "voice"}
        patched = next(t for t in body["turns"] if t["turn_id"] == turn_id)
        assert patched["client_ttfa_ms"] == 42.0
        assert body["summary"]["ttfa_ms"]["p50"] is not None
        assert client.get("/api/health").json()["ok"] is True
