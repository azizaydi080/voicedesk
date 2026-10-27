"""Decode any audio file (mp3, wav, ...) to PCM16 mono 16 kHz with PyAV."""

from __future__ import annotations

from pathlib import Path

from voicedesk.providers.base import SAMPLE_RATE


def decode_to_pcm16(path: Path | str) -> bytes:
    import av

    out = bytearray()
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                out.extend(bytes(resampled.planes[0])[: resampled.samples * 2])
        for resampled in resampler.resample(None):
            out.extend(bytes(resampled.planes[0])[: resampled.samples * 2])
    return bytes(out)
