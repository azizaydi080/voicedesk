import asyncio
from typing import Any

from voicedesk.config import Settings
from voicedesk.pipeline.session import Sender, VoiceSession
from voicedesk.providers.base import Transcript
from voicedesk.providers.fakes import FakeSTT, FakeTTS, ScriptedLLM
from voicedesk.providers.registry import Providers
from tests.conftest import TODAY, silence, tone


class Recorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.audio: list[bytes] = []

    async def json(self, msg: dict[str, Any]) -> None:
        self.events.append(msg)

    async def audio_frame(self, meta: dict[str, Any], data: bytes) -> None:
        self.events.append(meta)
        self.audio.append(data)

    def types(self) -> list[str]:
        return [e["type"] if e["type"] != "vad" else e["event"] for e in self.events]


def make_session(settings: Settings, transcripts: list[Transcript], token_delay: float = 0.0):
    rec = Recorder()
    providers = Providers(FakeSTT(transcripts), ScriptedLLM(token_delay), FakeTTS())
    session = VoiceSession(providers, settings, Sender(rec.json, rec.audio_frame), today=TODAY)
    return session, rec


async def test_voice_turn_end_to_end(settings: Settings) -> None:
    session, rec = make_session(settings, [Transcript("Is a dentiste free tomorrow?", "fr", 0.9)])
    await session.handle_audio(silence(300) + tone(900) + silence(700))
    await session.wait_idle()
    t = rec.types()
    assert t[:3] == ["speech_start", "speech_end", "transcript"]
    assert "tool_call" in t and "tool_result" in t and "audio" in t and t[-1] == "turn_end"
    final = next(e for e in rec.events if e["type"] == "transcript")
    assert final["language"] == "en"  # text evidence beats the STT's guess
    m = session.completed_turns[0]
    assert m.tools == ["find_slots"]
    assert m.input_mode == "voice"
    assert all(
        v is not None and v >= 0 for v in (m.stt_ms, m.llm_ttft_ms, m.tts_ttfb_ms, m.ttfa_ms)
    )
    assert m.ttfa_ms <= m.turn_ms  # type: ignore[operator]
    assert rec.audio and rec.audio[0].startswith(b"AUDIO[en]")


async def test_typed_turn_and_language_switch(settings: Settings) -> None:
    session, rec = make_session(settings, [])
    await session.handle_text("Bonjour, est-ce que la clinique est ouverte demain ?")
    await session.wait_idle()
    await session.handle_text("مرحبا")
    await session.wait_idle()
    langs = [e["language"] for e in rec.events if e["type"] == "transcript"]
    assert langs == ["fr", "ar"]


async def test_barge_in_cancels_speaking_turn(settings: Settings) -> None:
    session, rec = make_session(
        settings, [Transcript("hello there"), Transcript("cancel booking 1001")], token_delay=0.05
    )
    await session.handle_audio(silence(300) + tone(600) + silence(700))
    await asyncio.sleep(0.6)  # agent is mid-answer, audio already sent
    await session.handle_audio(tone(600) + silence(700))
    await session.wait_idle()
    t = rec.types()
    assert "interrupt" in t
    assert next(i for i, e in enumerate(rec.events) if e["type"] == "audio") < t.index("interrupt")
    first, second = session.completed_turns
    assert second.transcript == "cancel booking 1001"
    assert first.interrupted and not second.interrupted
    assert second.tools == ["cancel_appointment"]
    assert any("[interrupted by caller]" in m.content for m in session.agent.history)


async def test_speech_before_any_answer_is_merged(settings: Settings) -> None:
    settings.vad_end_silence_ms = 200
    session, rec = make_session(
        settings, [Transcript("I want to cancel"), Transcript("booking 1003")]
    )
    session.providers.stt.delay_s = 0.3  # type: ignore[attr-defined]
    await session.handle_audio(silence(300) + tone(500) + silence(250))
    await session.handle_audio(tone(500) + silence(400))  # caller keeps talking while STT runs
    await session.wait_idle()
    finals = [e["text"] for e in rec.events if e["type"] == "transcript" and e.get("final")]
    assert finals[-1] == "I want to cancel booking 1003"
    assert session.completed_turns[-1].tools == ["cancel_appointment"]
    user_msgs = [m for m in session.agent.history if m.role == "user"]
    assert len(user_msgs) == 1  # the abandoned partial turn was rolled back


async def test_empty_transcript_is_ignored(settings: Settings) -> None:
    session, rec = make_session(settings, [Transcript("")])
    await session.handle_audio(silence(200) + tone(300) + silence(700))
    await session.wait_idle()
    assert rec.events[-1] == {
        "type": "turn_end",
        "turn_id": rec.events[-1]["turn_id"],
        "empty": True,
    }
    assert session.completed_turns == []
