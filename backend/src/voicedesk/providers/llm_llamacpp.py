"""In-process local LLM via llama-cpp-python (GGUF), no server and no API key.

Targets Qwen2.5-Instruct's ChatML + Hermes-style tool calling. We render the prompt and
parse ``<tool_call>`` blocks ourselves so text deltas can be streamed to TTS while a tool
call is never spoken aloud. llama-cpp-python reuses the KV cache for the longest common
token prefix with the previous request; because the system prompt and history are
stable, each turn only pays prefill for the new tokens. We report that as
``cached_tokens``.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from voicedesk.providers.base import (
    LLMDone,
    LLMEvent,
    Message,
    TextDelta,
    ToolCall,
    ToolCallsReady,
    ToolSpec,
)

_OPEN, _CLOSE = "<tool_call>", "</tool_call>"

_TOOLS_PREAMBLE = """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def render_chatml(messages: list[Message], tools: list[ToolSpec]) -> str:
    parts: list[str] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.role == "system":
            content = m.content
            if i == 0 and tools:
                specs = "\n".join(json.dumps(t.as_openai(), ensure_ascii=False) for t in tools)
                content += _TOOLS_PREAMBLE.format(tools=specs)
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif m.role == "user":
            parts.append(f"<|im_start|>user\n{m.content}<|im_end|>")
        elif m.role == "assistant":
            body = m.content
            for c in m.tool_calls:
                call = json.dumps({"name": c.name, "arguments": c.arguments}, ensure_ascii=False)
                sep = "\n" if body else ""
                body += f"{sep}{_OPEN}\n{call}\n{_CLOSE}"
            parts.append(f"<|im_start|>assistant\n{body}<|im_end|>")
        elif m.role == "tool":
            responses = []
            while i < len(messages) and messages[i].role == "tool":
                responses.append(f"<tool_response>\n{messages[i].content}\n</tool_response>")
                i += 1
            parts.append("<|im_start|>user\n" + "\n".join(responses) + "<|im_end|>")
            continue
        i += 1
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


class ToolCallStreamParser:
    """Separates speakable text from ``<tool_call>`` blocks in a token stream."""

    def __init__(self) -> None:
        self._buf = ""
        self._in_call = False
        self.raw_calls: list[str] = []

    def feed(self, text: str) -> str:
        self._buf += text
        out = ""
        while self._buf:
            if self._in_call:
                end = self._buf.find(_CLOSE)
                if end < 0:
                    return out
                self.raw_calls.append(self._buf[:end].strip())
                self._buf = self._buf[end + len(_CLOSE) :]
                self._in_call = False
                continue
            start = self._buf.find(_OPEN)
            if start >= 0:
                out += self._buf[:start]
                self._buf = self._buf[start + len(_OPEN) :]
                self._in_call = True
                continue
            # Hold back a trailing fragment that might be the start of "<tool_call>".
            hold = next(
                (
                    k
                    for k in range(min(len(_OPEN), len(self._buf)), 0, -1)
                    if _OPEN.startswith(self._buf[-k:])
                ),
                0,
            )
            out += self._buf[: len(self._buf) - hold]
            self._buf = self._buf[len(self._buf) - hold :]
            return out
        return out

    def finish(self) -> tuple[str, list[ToolCall]]:
        tail = "" if self._in_call else self._buf
        if self._in_call and self._buf.strip():
            self.raw_calls.append(self._buf.strip())  # unterminated final call
        calls: list[ToolCall] = []
        for raw in self.raw_calls:
            try:
                obj = json.loads(raw)
                calls.append(
                    ToolCall(
                        f"call_{uuid.uuid4().hex[:8]}",
                        str(obj["name"]),
                        dict(obj.get("arguments") or {}),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return tail, calls


class LlamaCppLLM:
    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 220,
    ) -> None:
        from llama_cpp import Llama

        self.name = f"llamacpp:{model_path.name}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm: Any = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads or max(1, (os.cpu_count() or 4) // 2),
            verbose=False,
        )
        self._lock = asyncio.Lock()

    async def warmup(
        self, messages: list[Message] | None = None, tools: list[ToolSpec] | None = None
    ) -> None:
        """Prefill the stable prefix (system prompt + tool schemas) into the KV cache, so the
        first real turn only pays for the caller's words. On CPU this is the difference
        between a ~12 s and a ~1-2 s first response."""
        prefix = render_chatml(messages or [Message("system", "")], tools or [])
        prefix = prefix.removesuffix("<|im_start|>assistant\n") + "\n<|im_start|>user\n"
        async with self._lock:
            await asyncio.to_thread(self._llm.create_completion, prefix, max_tokens=1)

    def _common_prefix(self, tokens: list[int]) -> int:
        prev = list(self._llm._input_ids)  # tokens currently in the KV cache
        n = 0
        for a, b in zip(prev, tokens, strict=False):
            if a != b:
                break
            n += 1
        return n

    async def stream(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        prompt = render_chatml(messages, tools)
        async with self._lock:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            stop = threading.Event()
            tokens = self._llm.tokenize(prompt.encode(), add_bos=False, special=True)
            cached = self._common_prefix(tokens)

            def worker() -> None:
                n = 0
                try:
                    for chunk in self._llm.create_completion(
                        prompt,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        stop=["<|im_end|>", "<|endoftext|>"],
                        stream=True,
                    ):
                        if stop.is_set():
                            break
                        n += 1
                        loop.call_soon_threadsafe(
                            queue.put_nowait, ("text", chunk["choices"][0]["text"])
                        )
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, ("done", n))

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            parser = ToolCallStreamParser()
            completion_tokens = 0
            try:
                while True:
                    kind, payload = await queue.get()
                    if kind == "text":
                        speakable = parser.feed(payload)
                        if speakable:
                            yield TextDelta(speakable)
                    elif kind == "error":
                        raise RuntimeError(f"llama.cpp generation failed: {payload}")
                    else:
                        completion_tokens = int(payload)
                        break
            finally:
                stop.set()
                await asyncio.to_thread(thread.join)
            tail, calls = parser.finish()
            if tail.strip():
                yield TextDelta(tail)
            if calls:
                yield ToolCallsReady(calls)
            yield LLMDone(len(tokens), completion_tokens, cached)
