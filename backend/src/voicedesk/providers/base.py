"""Provider-agnostic interfaces for speech-to-text, the language model and text-to-speech.

Every concrete provider (local or hosted) implements one of these protocols, so the
pipeline, the evals and the benchmarks never import a vendor SDK directly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Language = Literal["ar", "fr", "en"]
SUPPORTED_LANGUAGES: tuple[Language, ...] = ("ar", "fr", "en")

SAMPLE_RATE = 16_000  # PCM16 mono, little-endian, everywhere inside the pipeline


# --------------------------------------------------------------------------- STT


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str | None = None
    language_probability: float | None = None
    audio_seconds: float = 0.0


PartialCallback = Callable[[str], Awaitable[None]]


class STTStream(Protocol):
    """One utterance. Streaming providers transcribe while audio arrives; batch
    providers buffer and do all the work in :meth:`finish`."""

    async def send(self, pcm16: bytes) -> None: ...

    async def finish(self) -> Transcript: ...

    async def abort(self) -> None: ...


@runtime_checkable
class STTProvider(Protocol):
    name: str

    async def open_stream(self, on_partial: PartialCallback | None = None) -> STTStream: ...

    async def warmup(self) -> None: ...


# --------------------------------------------------------------------------- LLM


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def as_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallsReady:
    calls: list[ToolCall]


@dataclass(frozen=True, slots=True)
class LLMDone:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    throttled_ms: float = 0.0  # time spent backing off from provider rate limits


LLMEvent = TextDelta | ToolCallsReady | LLMDone


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def stream(self, messages: list[Message], tools: list[ToolSpec]) -> AsyncIterator[LLMEvent]:
        """Yield text deltas as they are generated. If the model decides to call tools,
        yield a single :class:`ToolCallsReady` (text before it may be empty), then
        :class:`LLMDone`."""
        ...

    async def warmup(
        self, messages: list[Message] | None = None, tools: list[ToolSpec] | None = None
    ) -> None:
        """Pay one-off costs (connections, model load, prefill of a stable prompt prefix)."""
        ...


# --------------------------------------------------------------------------- TTS


AudioFormat = Literal["mp3", "pcm16"]


@runtime_checkable
class TTSProvider(Protocol):
    name: str
    audio_format: AudioFormat

    def synthesize(self, text: str, language: Language) -> AsyncIterator[bytes]:
        """Yield encoded audio bytes for ``text`` as soon as the provider produces them."""
        ...
