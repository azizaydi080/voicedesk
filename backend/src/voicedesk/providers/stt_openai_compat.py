"""Hosted Whisper over an OpenAI-compatible ``/audio/transcriptions`` endpoint (Groq,
OpenAI, a local whisper server...). Batch per utterance like local Whisper, but the
compute runs on the provider's hardware: on Groq, ``whisper-large-v3-turbo`` returns in a
few hundred milliseconds for a short utterance."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field
from typing import Any

import httpx

from voicedesk.providers.base import SAMPLE_RATE, PartialCallback, Transcript

# Language names as returned by verbose_json -> our codes.
_LANG = {"arabic": "ar", "french": "fr", "english": "en", "ar": "ar", "fr": "fr", "en": "en"}


def pcm16_to_wav(pcm16: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16)
    return buf.getvalue()


@dataclass
class _Stream:
    stt: OpenAICompatSTT
    _audio: bytearray = field(default_factory=bytearray)

    async def send(self, pcm16: bytes) -> None:
        self._audio.extend(pcm16)

    async def finish(self) -> Transcript:
        return await self.stt.transcribe(bytes(self._audio))

    async def abort(self) -> None:
        self._audio.clear()


class OpenAICompatSTT:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = f"openai_stt:{model}"
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=httpx.Timeout(20.0)
        )

    async def open_stream(self, on_partial: PartialCallback | None = None) -> _Stream:
        return _Stream(self)

    async def warmup(self) -> None:
        await self.transcribe(bytes(SAMPLE_RATE))  # opens the TLS connection

    async def transcribe(self, pcm16: bytes) -> Transcript:
        seconds = len(pcm16) / (2 * SAMPLE_RATE)
        if seconds < 0.2:
            return Transcript("", None, None, seconds)
        resp = await self._client.post(
            "/audio/transcriptions",
            files={"file": ("utterance.wav", pcm16_to_wav(pcm16), "audio/wav")},
            data={"model": self.model, "response_format": "verbose_json", "temperature": "0"},
        )
        resp.raise_for_status()
        body: dict[str, Any] = resp.json()
        lang = _LANG.get(str(body.get("language", "")).lower())
        return Transcript(str(body.get("text", "")).strip(), lang, 0.9 if lang else None, seconds)
