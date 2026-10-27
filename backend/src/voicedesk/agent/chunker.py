"""Splits streamed LLM tokens into speakable chunks for incremental TTS.

The first chunk is cut at the earliest clause boundary (comma included) once it has a
few characters, because time-to-first-audio is dominated by how soon the first TTS
request can start. Later chunks wait for sentence ends so prosody stays natural.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?؟。…]+[\"')\]]*\s")
_CLAUSE_END = re.compile(r"[,;:،؛]\s|[.!?؟。…]+[\"')\]]*\s")


class SentenceChunker:
    def __init__(self, first_min_chars: int = 12, min_chars: int = 40) -> None:
        self.first_min_chars = first_min_chars
        self.min_chars = min_chars
        self._buf = ""
        self._emitted = 0

    def push(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while True:
            chunk = self._take()
            if chunk is None:
                return out
            out.append(chunk)

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        if rest:
            self._emitted += 1
            return [rest]
        return []

    def _take(self) -> str | None:
        first = self._emitted == 0
        pattern = _CLAUSE_END if first else _SENTENCE_END
        minimum = self.first_min_chars if first else self.min_chars
        for m in pattern.finditer(self._buf):
            if m.end() >= minimum:
                chunk, self._buf = self._buf[: m.end()].strip(), self._buf[m.end() :]
                self._emitted += 1
                return chunk
        return None
