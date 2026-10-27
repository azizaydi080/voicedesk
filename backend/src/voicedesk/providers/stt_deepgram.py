"""Streaming speech-to-text over Deepgram's live WebSocket API.

Audio is forwarded while the caller is still speaking, so when our endpointer fires only
the ``Finalize`` round trip remains. Requires ``VOICEDESK_DEEPGRAM_API_KEY``.
Use ``VOICEDESK_DEEPGRAM_LANGUAGE=ar`` for Arabic-only lines: as of writing, Deepgram's
``multi`` code-switching mode does not include Arabic.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any
from urllib.parse import urlencode

from voicedesk.providers.base import SAMPLE_RATE, PartialCallback, Transcript

_URL = "wss://api.deepgram.com/v1/listen"


class _DeepgramStream:
    def __init__(self, ws: Any, on_partial: PartialCallback | None) -> None:
        self._ws = ws
        self._on_partial = on_partial
        self._final: list[str] = []
        self._language: str | None = None
        self._confidence: float | None = None
        self._bytes = 0
        self._finalized = asyncio.Event()
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        async for raw in self._ws:
            msg = json.loads(raw)
            if msg.get("type") != "Results":
                continue
            alt = msg["channel"]["alternatives"][0]
            langs = alt.get("languages") or []
            if langs:
                self._language = langs[0]
            if msg.get("is_final"):
                if alt.get("transcript"):
                    self._final.append(alt["transcript"])
                    self._confidence = alt.get("confidence")
                if msg.get("from_finalize"):
                    self._finalized.set()
            elif self._on_partial and alt.get("transcript"):
                await self._on_partial(" ".join([*self._final, alt["transcript"]]))

    async def send(self, pcm16: bytes) -> None:
        self._bytes += len(pcm16)
        await self._ws.send(pcm16)

    async def finish(self) -> Transcript:
        await self._ws.send(json.dumps({"type": "Finalize"}))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._finalized.wait(), timeout=3.0)
        await self.abort()
        return Transcript(
            " ".join(self._final).strip(),
            self._language,
            self._confidence,
            self._bytes / (2 * SAMPLE_RATE),
        )

    async def abort(self) -> None:
        with contextlib.suppress(Exception):
            await self._ws.send(json.dumps({"type": "CloseStream"}))
            await self._ws.close()
        self._reader.cancel()


class DeepgramSTT:
    def __init__(self, api_key: str, model: str = "nova-3", language: str = "multi") -> None:
        self.name = f"deepgram:{model}:{language}"
        self._key = api_key
        self._params = urlencode(
            {
                "model": model,
                "language": language,
                "encoding": "linear16",
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "interim_results": "true",
                "punctuate": "true",
                "smart_format": "true",
            }
        )

    async def open_stream(self, on_partial: PartialCallback | None = None) -> _DeepgramStream:
        import websockets

        ws = await websockets.connect(
            f"{_URL}?{self._params}", additional_headers={"Authorization": f"Token {self._key}"}
        )
        return _DeepgramStream(ws, on_partial)

    async def warmup(self) -> None:
        return None
