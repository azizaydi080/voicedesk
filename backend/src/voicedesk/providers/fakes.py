"""Deterministic test doubles. They exercise the pipeline's plumbing (streaming, tool
dispatch, barge-in, metrics) without any model; they say nothing about model quality,
which the evals measure with real providers."""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from voicedesk.providers.base import (
    AudioFormat,
    LLMDone,
    LLMEvent,
    Message,
    PartialCallback,
    TextDelta,
    ToolCall,
    ToolCallsReady,
    ToolSpec,
    Transcript,
)


@dataclass
class FakeSTTStream:
    transcript_for: Callable[[bytes], Transcript]
    delay_s: float
    _audio: bytearray = field(default_factory=bytearray)

    async def send(self, pcm16: bytes) -> None:
        self._audio.extend(pcm16)

    async def finish(self) -> Transcript:
        await asyncio.sleep(self.delay_s)
        return self.transcript_for(bytes(self._audio))

    async def abort(self) -> None:
        self._audio.clear()


class FakeSTT:
    """Returns queued transcripts in order (one per utterance)."""

    name = "fake"

    def __init__(self, transcripts: list[Transcript] | None = None, delay_s: float = 0.01) -> None:
        self.queue = list(transcripts or [])
        self.delay_s = delay_s

    def _next(self, audio: bytes) -> Transcript:
        if self.queue:
            return self.queue.pop(0)
        return Transcript("", None, None, len(audio) / 32000)

    async def open_stream(self, on_partial: PartialCallback | None = None) -> FakeSTTStream:
        return FakeSTTStream(self._next, self.delay_s)

    async def warmup(self) -> None:
        return None


class ScriptedLLM:
    """Keyword rules -> tool calls, then a templated answer that echoes the tool output."""

    name = "scripted"

    def __init__(self, token_delay_s: float = 0.005) -> None:
        self.token_delay_s = token_delay_s

    async def warmup(
        self, messages: list[Message] | None = None, tools: list[ToolSpec] | None = None
    ) -> None:
        return None

    async def stream(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        last = messages[-1]
        if last.role == "tool":
            reply = f"Here is what I found: {last.content}. Anything else?"
        else:
            text = last.content.lower()
            call = None
            if tools and (m := re.search(r"\b(\d{4})\b", text)) and "cancel" in text:
                call = ToolCall(
                    uuid.uuid4().hex[:8], "cancel_appointment", {"booking_ref": m.group(1)}
                )
            elif tools and (m := re.search(r"\b(\d{4})\b", text)):
                call = ToolCall(
                    uuid.uuid4().hex[:8], "get_appointment", {"booking_ref": m.group(1)}
                )
            elif tools and ("dentist" in text or "dentiste" in text):
                call = ToolCall(
                    uuid.uuid4().hex[:8], "find_slots", {"specialty": "dentist", "date": "tomorrow"}
                )
            if call:
                yield ToolCallsReady([call])
                yield LLMDone(len(text), 0, 0)
                return
            reply = (
                "Hello, this is Ennour Clinic. How can I help you today? I can book appointments."
            )
        for word in reply.split(" "):
            await asyncio.sleep(self.token_delay_s)
            yield TextDelta(word + " ")
        yield LLMDone(None, len(reply.split()), None)


class FakeTTS:
    name = "fake"
    audio_format: AudioFormat = "mp3"

    def __init__(self, delay_s: float = 0.01) -> None:
        self.delay_s = delay_s
        self.spoken: list[str] = []

    async def synthesize(self, text: str, language: str) -> AsyncIterator[bytes]:
        self.spoken.append(text)
        await asyncio.sleep(self.delay_s)
        yield f"AUDIO[{language}]:{text}".encode()
