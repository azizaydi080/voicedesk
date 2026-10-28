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
    """ElevenLabs streaming endpoint. ``eleven_flash_v2_5`` (the default) covers Arabic, French
    and English with one voice at the lowest latency; a voice per language is optional.
    Requires ``VOICEDESK_ELEVENLABS_API_KEY``."""

    name = "elevenlabs"
    audio_format: AudioFormat = "mp3"
    # Same encoding as edge-tts (24 kHz, 48 kbps mono MP3), so the browser path is unchanged.
    output_format = "mp3_24000_48"

    def __init__(
        self,
        api_key: str,
        voices: dict[Language, str],
        model: str = "eleven_flash_v2_5",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.voices = voices
        self.model = model
        self._client = client or httpx.AsyncClient(
            base_url="https://api.elevenlabs.io/v1", headers={"xi-api-key": api_key}, timeout=30.0
        )

    async def synthesize(self, text: str, language: Language) -> AsyncIterator[bytes]:
        body = {"text": text, "model_id": self.model}
        # language_code pins pronunciation and text normalisation; multilingual_v2 rejects it.
        if not self.model.startswith("eleven_multilingual_v2"):
            body["language_code"] = language
        async with self._client.stream(
            "POST",
            f"/text-to-speech/{self.voices[language]}/stream",
            params={"output_format": self.output_format},
            json=body,
        ) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread()).decode(errors="replace")[:500]
                raise RuntimeError(f"ElevenLabs HTTP {resp.status_code}: {detail}")
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk
