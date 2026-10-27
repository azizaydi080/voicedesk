"""Runs scenarios through the real agent (and optionally real STT on audio fixtures)."""

from __future__ import annotations

import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from voicedesk.agent.agent import Agent, AgentLLMRound, AgentText, AgentToolEnd, AgentToolStart
from voicedesk.domain.clinic import ClinicStore
from voicedesk.domain.tools import ToolRunner
from voicedesk.evals.scoring import Scenario, Suite, cer, score_answer, score_success, score_tools
from voicedesk.language import detect_language
from voicedesk.metrics.store import percentile
from voicedesk.providers.base import Language, LLMProvider, STTProvider, ToolCall


@dataclass
class TurnResult:
    scenario: str
    group: str
    index: int
    expected_lang: str
    user_text: str
    transcript: str | None = None
    stt_language: str | None = None
    stt_ms: float | None = None
    cer: float | None = None
    detected_lang: str | None = None
    reply: str = ""
    reply_lang: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tools_ok: bool | None = None
    answer_ok: bool | None = None
    llm_ttft_ms: float | None = None
    agent_ms: float | None = None
    error: str | None = None

    @property
    def lang_ok(self) -> bool:
        return self.detected_lang == self.expected_lang

    @property
    def reply_lang_ok(self) -> bool:
        return self.reply_lang == self.expected_lang


@dataclass
class ScenarioResult:
    id: str
    group: str
    turns: list[TurnResult]
    state_ok: bool | None

    @property
    def success(self) -> bool:
        tools = all(t.tools_ok is not False and t.answer_ok is not False for t in self.turns)
        langs = all(t.lang_ok and (t.reply_lang_ok or not t.reply) for t in self.turns)
        return (
            tools and langs and self.state_ok is not False and not any(t.error for t in self.turns)
        )


def fixture_path(audio_dir: Path, scenario_id: str, index: int) -> Path:
    return audio_dir / f"{scenario_id}_{index}.wav"


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16_000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return w.readframes(w.getnframes())


async def run_scenario(
    sc: Scenario,
    suite: Suite,
    llm: LLMProvider | None,
    stt: STTProvider | None,
    audio_dir: Path,
) -> ScenarioResult:
    store = ClinicStore(suite.today)
    agent = Agent(llm, ToolRunner(store), suite.today) if llm else None
    previous: Language | None = None
    results: list[TurnResult] = []
    for i, turn in enumerate(sc.turns):
        r = TurnResult(sc.id, sc.group, i, turn.lang, turn.user)
        text, stt_lang, stt_prob = turn.user, None, None
        try:
            if stt is not None:
                stream = await stt.open_stream()
                await stream.send(read_wav(fixture_path(audio_dir, sc.id, i)))
                t0 = time.perf_counter()
                tr = await stream.finish()
                r.stt_ms = round((time.perf_counter() - t0) * 1000, 1)
                text, stt_lang, stt_prob = tr.text, tr.language, tr.language_probability
                r.transcript, r.stt_language = tr.text, tr.language
                r.cer = round(cer(turn.user, tr.text), 4)
            guess = detect_language(text, stt_lang, stt_prob, previous)
            previous = guess.language
            r.detected_lang = guess.language
            if agent is not None:
                calls: list[ToolCall] = []
                reply: list[str] = []
                t0 = time.perf_counter()
                async for ev in agent.run_turn(text, guess.language):
                    if isinstance(ev, AgentText):
                        reply.append(ev.text)
                    elif isinstance(ev, AgentToolStart):
                        calls.append(ToolCall(ev.call_id, ev.name, dict(ev.arguments)))
                    elif isinstance(ev, AgentToolEnd):
                        r.tool_calls.append(
                            {
                                "name": ev.result.call.name,
                                "arguments": ev.result.call.arguments,
                                "ok": ev.result.ok,
                            }
                        )
                    elif isinstance(ev, AgentLLMRound) and ev.round == 0 and ev.ttft_ms:
                        r.llm_ttft_ms = round(ev.ttft_ms, 1)
                r.agent_ms = round((time.perf_counter() - t0) * 1000, 1)
                r.reply = "".join(reply).strip()
                r.reply_lang = detect_language(r.reply).language if r.reply else None
                r.tools_ok = score_tools(turn, calls, suite.today)
                r.answer_ok = score_answer(turn, r.reply)
        except Exception as exc:  # a crash is a failed turn, not a failed run
            r.error = f"{type(exc).__name__}: {exc}"
        results.append(r)
    state_ok = score_success(sc.success, store) if (agent and sc.success) else None
    return ScenarioResult(sc.id, sc.group, results, state_ok)


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def aggregate(results: list[ScenarioResult], with_agent: bool) -> dict[str, Any]:
    turns = [t for s in results for t in s.turns]
    summary: dict[str, Any] = {
        "scenarios": len(results),
        "turns": len(turns),
        "language_id_accuracy": _rate([t.lang_ok for t in turns]),
    }
    cers = [t.cer for t in turns if t.cer is not None]
    if cers:
        summary["stt_cer_mean"] = round(mean(cers), 3)
        summary["stt_ms_p50"] = percentile([t.stt_ms for t in turns if t.stt_ms], 50)
    if with_agent:
        summary |= {
            "tool_call_accuracy": _rate([t.tools_ok for t in turns if t.tools_ok is not None]),
            "faq_answer_accuracy": _rate([t.answer_ok for t in turns if t.answer_ok is not None]),
            "reply_language_accuracy": _rate([t.reply_lang_ok for t in turns if t.reply]),
            "task_success_rate": _rate([s.success for s in results]),
            "errors": sum(1 for t in turns if t.error),
            "llm_ttft_ms_p50": percentile([t.llm_ttft_ms for t in turns if t.llm_ttft_ms], 50),
            "agent_turn_ms_p50": percentile([t.agent_ms for t in turns if t.agent_ms], 50),
        }
    groups: dict[str, Any] = {}
    for g in dict.fromkeys(s.group for s in results):
        sub = [s for s in results if s.group == g]
        gt = [t for s in sub for t in s.turns]
        row: dict[str, Any] = {"turns": len(gt), "language_id": _rate([t.lang_ok for t in gt])}
        if cers:
            row["cer"] = round(mean(t.cer for t in gt if t.cer is not None), 3)
        if with_agent:
            row["tool_calls"] = _rate([t.tools_ok for t in gt if t.tools_ok is not None])
            row["task_success"] = _rate([s.success for s in sub])
        groups[g] = row
    summary["by_group"] = groups
    return summary


def result_rows(results: list[ScenarioResult]) -> list[dict[str, Any]]:
    return [
        {
            "id": s.id,
            "success": s.success,
            "state_ok": s.state_ok,
            "turns": [asdict(t) | {"lang_ok": t.lang_ok} for t in s.turns],
        }
        for s in results
    ]
