"""Text-to-speech providers. Both stream MP3 bytes as they are synthesised."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from voicedesk.providers.base import AudioFormat, Language


class EdgeTTS:
    """Microsoft Edge's online neural voices via ``edge-tts``: free, no key, needs internet."""

    name = "edge"
    audio_format: AudioFormat = "mp3"

    def __init__(self, voices: dict[Language, str]) -> None:
        self.voices = voices

    async def synthesize(self, text: str, language: Language) -> AsyncIterator[bytes]:
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voices[language])
        async for chunk in communicate.stream():
            if chunk["type"] == "audio" and chunk.get("data"):
                yield chunk["data"]


class ElevenLabsTTS:
    """ElevenLabs streaming endpoint; ``eleven_flash_v2_5`` covers Arabic, French and English
    with one voice. Requires ``VOICEDESK_ELEVENLABS_API_KEY``."""

    name = "elevenlabs"
    audio_format: AudioFormat = "mp3"

    def __init__(self, api_key: str, voice_id: str, model: str = "eleven_flash_v2_5") -> None:
        self.voice_id = voice_id
        self.model = model
        self._client = httpx.AsyncClient(
            base_url="https://api.elevenlabs.io/v1", headers={"xi-api-key": api_key}, timeout=30.0
        )

    async def synthesize(self, text: str, language: Language) -> AsyncIterator[bytes]:
        body = {"text": text, "model_id": self.model, "language_code": language}
        async with self._client.stream(
            "POST",
            f"/text-to-speech/{self.voice_id}/stream",
            params={"output_format": "mp3_22050_32", "optimize_streaming_latency": 3},
            json=body,
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk
