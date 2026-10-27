"""Local speech-to-text with faster-whisper (CTranslate2, int8 on CPU).

Whisper is not a streaming model: audio is buffered while the caller speaks and
transcribed when the endpointer fires, so STT latency is paid after end of speech. Use
:class:`~voicedesk.providers.stt_deepgram.DeepgramSTT` for true streaming.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from voicedesk.providers.base import SAMPLE_RATE, SUPPORTED_LANGUAGES, PartialCallback, Transcript

# Biases decoding toward the domain vocabulary; cheap and noticeably helps names/specialties.
_PROMPTS = {
    "ar": "مرحبا، أريد موعدا في عيادة النور عند طبيب الأسنان.",
    "fr": "Bonjour, je voudrais un rendez-vous à la clinique Ennour chez le dentiste.",
    "en": "Hello, I would like an appointment at Ennour Clinic with the dentist.",
}


@dataclass
class _WhisperStream:
    stt: WhisperSTT
    _audio: bytearray = field(default_factory=bytearray)

    async def send(self, pcm16: bytes) -> None:
        self._audio.extend(pcm16)

    async def finish(self) -> Transcript:
        return await asyncio.to_thread(self.stt.transcribe_pcm, bytes(self._audio))

    async def abort(self) -> None:
        self._audio.clear()


class WhisperSTT:
    def __init__(
        self, model_size: str = "small", compute_type: str = "int8", beam_size: int = 1
    ) -> None:
        from faster_whisper import WhisperModel

        self.name = f"whisper:{model_size}"
        self.beam_size = beam_size
        self._model: Any = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        self._lock = asyncio.Lock()

    async def open_stream(self, on_partial: PartialCallback | None = None) -> _WhisperStream:
        return _WhisperStream(self)

    async def warmup(self) -> None:
        await asyncio.to_thread(self.transcribe_pcm, bytes(SAMPLE_RATE))

    def transcribe_pcm(self, pcm16: bytes) -> Transcript:
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        seconds = len(audio) / SAMPLE_RATE
        if seconds < 0.2:
            return Transcript("", None, None, seconds)
        # One encoder pass for language ID + decoding in the common case. If Whisper picks a
        # language we do not serve (Maghrebi Arabic is often labelled Maltese or Persian),
        # re-decode forced to the most likely supported language.
        text, info = self._decode(audio, None)
        probs = dict(info.all_language_probs or [])
        lang = info.language
        if lang not in SUPPORTED_LANGUAGES:
            lang = max(SUPPORTED_LANGUAGES, key=lambda code: probs.get(code, 0.0))
            text, _ = self._decode(audio, lang)
        return Transcript(text, lang, float(probs.get(lang, info.language_probability)), seconds)

    def _decode(self, audio: np.ndarray, language: str | None) -> tuple[str, Any]:
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=_PROMPTS.get(language or ""),
            without_timestamps=True,
        )
        return " ".join(s.text.strip() for s in segments).strip(), info
