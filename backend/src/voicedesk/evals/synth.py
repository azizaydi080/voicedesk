"""Synthesises the audio fixtures for the audio-mode evals with edge-tts.

Fixtures are committed so evals and benchmarks run offline and reproducibly. They are
synthetic speech: cleaner than a real phone line, so STT numbers are an upper bound.
"""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

from voicedesk.audio.decode import decode_to_pcm16
from voicedesk.evals.runner import fixture_path
from voicedesk.evals.scoring import Suite

DEFAULT_VOICES = {"ar": "ar-SA-HamedNeural", "fr": "fr-FR-HenriNeural", "en": "en-GB-RyanNeural"}


def write_wav(path: Path, pcm16: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(pcm16)


async def synthesize_fixtures(suite: Suite, audio_dir: Path, force: bool = False) -> list[Path]:
    import edge_tts

    audio_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 (one-off CLI)
    written = []
    for sc in suite.scenarios:
        for i, turn in enumerate(sc.turns):
            out = fixture_path(audio_dir, sc.id, i)
            if out.exists() and not force:
                continue
            voice = turn.voice or DEFAULT_VOICES[turn.lang]
            with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
                await edge_tts.Communicate(turn.user, voice).save(tmp.name)
                pcm = decode_to_pcm16(tmp.name)
            # 300 ms of silence either side, like a caller pausing before and after speaking.
            pad = bytes(int(0.3 * 16_000) * 2)
            write_wav(out, pad + pcm + pad)
            written.append(out)
    return written
