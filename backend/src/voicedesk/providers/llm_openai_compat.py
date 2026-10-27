"""Any OpenAI-compatible chat completions endpoint: Groq, Gemini (OpenAI mode), OpenAI,
Ollama, vLLM or a llama.cpp server. Streams text and accumulates streamed tool calls."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from voicedesk.providers.base import (
    LLMDone,
    LLMEvent,
    Message,
    TextDelta,
    ToolCall,
    ToolCallsReady,
    ToolSpec,
)


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        msg: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": json.dumps(c.arguments, ensure_ascii=False),
                    },
                }
                for c in m.tool_calls
            ]
        if m.role == "tool":
            msg["tool_call_id"] = m.tool_call_id
        out.append(msg)
    return out


class OpenAICompatLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 220,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = f"openai_compat:{model}"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=httpx.Timeout(30.0)
        )

    async def warmup(
        self, messages: list[Message] | None = None, tools: list[ToolSpec] | None = None
    ) -> None:
        # Opens the TLS connection so the first real turn does not pay the handshake.
        with contextlib.suppress(httpx.HTTPError):
            await self._client.get("/models")

    async def stream(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = [t.as_openai() for t in tools]
        pending: dict[int, dict[str, str]] = {}
        usage: dict[str, Any] = {}
        async with self._client.stream("POST", "/chat/completions", json=body) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread()).decode(errors="replace")[:500]
                raise RuntimeError(f"LLM HTTP {resp.status_code}: {detail}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        yield TextDelta(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        slot = pending.setdefault(
                            tc.get("index", 0), {"id": "", "name": "", "args": ""}
                        )
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["args"] += fn.get("arguments") or ""
        if pending:
            calls = []
            for i, slot in sorted(pending.items()):
                try:
                    args = json.loads(slot["args"] or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": slot["args"]}
                calls.append(ToolCall(slot["id"] or f"call_{i}", slot["name"], args))
            yield ToolCallsReady(calls)
        details = usage.get("prompt_tokens_details") or {}
        yield LLMDone(
            usage.get("prompt_tokens"), usage.get("completion_tokens"), details.get("cached_tokens")
        )
