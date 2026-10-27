"""The conversational agent: an LLM tool-calling loop that streams its spoken answer."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import date

from voicedesk.agent.prompts import language_note, system_prompt
from voicedesk.domain.tools import TOOL_SPECS, ToolResult, ToolRunner
from voicedesk.providers.base import (
    Language,
    LLMDone,
    LLMProvider,
    Message,
    TextDelta,
    ToolCallsReady,
)


@dataclass(frozen=True, slots=True)
class AgentText:
    text: str


@dataclass(frozen=True, slots=True)
class AgentToolStart:
    name: str
    arguments: dict[str, object]
    call_id: str


@dataclass(frozen=True, slots=True)
class AgentToolEnd:
    result: ToolResult


@dataclass(frozen=True, slots=True)
class AgentLLMRound:
    round: int
    ttft_ms: float | None
    total_ms: float
    prompt_tokens: int | None
    cached_tokens: int | None


AgentEvent = AgentText | AgentToolStart | AgentToolEnd | AgentLLMRound


@dataclass
class Agent:
    llm: LLMProvider
    tools: ToolRunner
    today: date
    max_tool_rounds: int = 3
    history: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append(Message("system", system_prompt(self.today)))

    async def warmup(self) -> None:
        await self.llm.warmup(self.history[:1], TOOL_SPECS)

    async def run_turn(
        self, user_text: str, language: Language
    ) -> AsyncGenerator[AgentEvent, None]:
        """Yield events for one user turn. If the consumer stops early (barge-in), the
        partial answer is still committed to history, marked as interrupted."""
        self.history.append(Message("user", f"{user_text}\n\n({language_note(language)})"))
        spoken: list[str] = []
        finished = False
        try:
            for rnd in range(self.max_tool_rounds + 1):
                tools = TOOL_SPECS if rnd < self.max_tool_rounds else []
                t0 = time.perf_counter()
                ttft: float | None = None
                calls = None
                usage = LLMDone()
                async for ev in self.llm.stream(self.history, tools):
                    if isinstance(ev, TextDelta):
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        spoken.append(ev.text)
                        yield AgentText(ev.text)
                    elif isinstance(ev, ToolCallsReady):
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        calls = ev.calls
                    elif isinstance(ev, LLMDone):
                        usage = ev
                yield AgentLLMRound(
                    rnd,
                    ttft,
                    (time.perf_counter() - t0) * 1000,
                    usage.prompt_tokens,
                    usage.cached_tokens,
                )
                if not calls:
                    finished = True
                    return
                # Text the model said before deciding to call tools is kept in the transcript.
                self.history.append(Message("assistant", "".join(spoken), tool_calls=calls))
                spoken.clear()
                for call in calls:
                    yield AgentToolStart(call.name, dict(call.arguments), call.id)
                    result = self.tools.run(call, language)
                    self.history.append(
                        Message("tool", result.as_content(), tool_call_id=call.id, name=call.name)
                    )
                    yield AgentToolEnd(result)
            finished = True
        finally:
            text = "".join(spoken).strip()
            if text or not finished:
                suffix = "" if finished else " [interrupted by caller]"
                self.history.append(Message("assistant", text + suffix))
