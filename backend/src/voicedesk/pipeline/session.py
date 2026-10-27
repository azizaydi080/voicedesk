"""One caller's real-time session: VAD -> STT -> language ID -> agent -> chunked TTS.

Concurrency model (per turn, one asyncio task):

    producer: agent tokens -> SentenceChunker -> queue
    consumer: queue -> TTS (chunk by chunk) -> WebSocket

so the first clause is being synthesised while the model is still writing the rest.
A new utterance (barge-in) or a typed message cancels the running turn task; cancellation
propagates into the LLM stream and the TTS request, and the client drops queued audio.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from voicedesk.agent.agent import Agent, AgentLLMRound, AgentText, AgentToolEnd, AgentToolStart
from voicedesk.agent.chunker import SentenceChunker
from voicedesk.audio.vad import EnergyVAD, VadEvent
from voicedesk.config import Settings
from voicedesk.domain.clinic import ClinicStore
from voicedesk.domain.tools import ToolRunner
from voicedesk.language import detect_language
from voicedesk.metrics.store import MetricsStore, TurnMetrics
from voicedesk.providers.base import Language, STTStream, Transcript
from voicedesk.providers.registry import Providers

PREROLL_FRAMES = 15  # 300 ms kept before the VAD fires, so the first syllable is not lost


@dataclass(slots=True)
class _TurnState:
    """What an in-flight turn has done so far; read by :meth:`VoiceSession.interrupt`."""

    stt_task: asyncio.Future[Transcript] | None
    history_len: int
    user_text: str = ""
    audio_sent: bool = False
    tools_ran: bool = False


@dataclass(frozen=True, slots=True)
class Sender:
    json: Callable[[dict[str, Any]], Awaitable[None]]
    audio: Callable[[dict[str, Any], bytes], Awaitable[None]]  # announcement + binary frame


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


class VoiceSession:
    def __init__(
        self,
        providers: Providers,
        settings: Settings,
        send: Sender,
        metrics: MetricsStore | None = None,
        today: date | None = None,
        session_id: str | None = None,
    ) -> None:
        self.providers = providers
        self.settings = settings
        self.send = send
        self.metrics = metrics
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.store = ClinicStore(today or settings.clinic_today or date.today())
        self.agent = Agent(
            providers.llm, ToolRunner(self.store), self.store.today, settings.max_tool_rounds
        )
        self.vad = EnergyVAD(
            settings.vad_threshold_db, settings.vad_start_ms, settings.vad_end_silence_ms
        )
        self.language: Language | None = None
        self.completed_turns: list[TurnMetrics] = []
        self._preroll: deque[bytes] = deque(maxlen=PREROLL_FRAMES)
        self._stt: STTStream | None = None
        self._turn: asyncio.Task[None] | None = None
        self._turn_state: _TurnState | None = None
        self._carryover = ""
        self._agent_busy = False

    # ------------------------------------------------------------------ inputs

    async def handle_audio(self, pcm16: bytes) -> None:
        for frame, event in self.vad.process(pcm16):
            if event is VadEvent.SPEECH_START:
                await self._on_speech_start()
            if self._stt is not None:
                await self._stt.send(frame)
            else:
                self._preroll.append(frame)
            if event is VadEvent.SPEECH_END:
                await self._on_speech_end()

    async def handle_text(self, text: str) -> None:
        await self.interrupt("new_input")
        self._start_turn(None, text, time.perf_counter())

    async def handle_control(self, msg: dict[str, Any]) -> None:
        if msg.get("type") == "playback_started" and self.metrics and msg.get("turn_id"):
            self.metrics.patch(str(msg["turn_id"]), client_ttfa_ms=msg.get("client_ttfa_ms"))
        elif msg.get("type") == "playback_ended" and (self._turn is None or self._turn.done()):
            self._set_busy(False)
        elif msg.get("type") == "interrupt":
            await self.interrupt("client")

    async def close(self) -> None:
        await self.interrupt("closed", notify=False)
        if self._stt is not None:
            await self._stt.abort()

    async def wait_idle(self) -> None:
        if self._turn is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn

    # ------------------------------------------------------------- turn taking

    async def _on_speech_start(self) -> None:
        if self._turn is not None and not self._turn.done():
            await self.interrupt("barge_in")
        await self.send.json({"type": "vad", "event": "speech_start"})
        self._stt = await self.providers.stt.open_stream(self._on_partial)
        for frame in self._preroll:
            await self._stt.send(frame)
        self._preroll.clear()

    async def _on_speech_end(self) -> None:
        stt, self._stt = self._stt, None
        await self.send.json({"type": "vad", "event": "speech_end"})
        if stt is not None:
            self._start_turn(stt, None, time.perf_counter())

    async def _on_partial(self, text: str) -> None:
        await self.send.json({"type": "transcript", "final": False, "text": text})

    async def interrupt(self, reason: str, notify: bool = True) -> None:
        task = self._turn
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        st = self._turn_state
        if st is not None and not st.audio_sent and not st.tools_ran:
            # The caller kept talking before we answered: treat both utterances as one.
            # This also covers a turn cancelled before its task ever ran.
            if not st.user_text and st.stt_task is not None:
                with contextlib.suppress(Exception):
                    st.user_text = f"{self._carryover} {(await st.stt_task).text}".strip()
            self._carryover = st.user_text or self._carryover
            del self.agent.history[st.history_len :]
        if notify:
            await self.send.json({"type": "interrupt", "reason": reason})

    def _set_busy(self, busy: bool) -> None:
        # While the agent talks, require a longer burst before treating sound as barge-in.
        self._agent_busy = busy
        self.vad.start_ms = self.settings.barge_in_min_ms if busy else self.settings.vad_start_ms

    def _start_turn(self, stt: STTStream | None, text: str | None, t_end: float) -> None:
        # STT runs as its own task so a barge-in during transcription still recovers it.
        stt_task = asyncio.ensure_future(stt.finish()) if stt is not None else None
        self._turn_state = _TurnState(stt_task, len(self.agent.history))
        self._turn = asyncio.create_task(self._run_turn(self._turn_state, text, t_end))

    # ------------------------------------------------------------------- a turn

    async def _run_turn(self, st: _TurnState, text: str | None, t_end: float) -> None:
        p = self.providers
        m = TurnMetrics(
            turn_id=uuid.uuid4().hex[:12],
            session_id=self.session_id,
            started_at=time.time(),
            input_mode="voice" if st.stt_task else "text",
            providers={"stt": p.stt.name, "llm": p.llm.name, "tts": p.tts.name},
            endpoint_silence_ms=self.settings.vad_end_silence_ms if st.stt_task else 0,
        )
        try:
            transcript = Transcript(text or "")
            if st.stt_task is not None:
                transcript = await asyncio.shield(st.stt_task)
                m.stt_ms, m.audio_seconds = _ms(t_end), round(transcript.audio_seconds, 2)
            user_text = st.user_text = f"{self._carryover} {transcript.text}".strip()
            self._carryover = ""
            if not user_text:
                await self.send.json({"type": "turn_end", "turn_id": m.turn_id, "empty": True})
                return
            guess = detect_language(
                user_text, transcript.language, transcript.language_probability, self.language
            )
            self.language = guess.language
            m.transcript, m.language, m.language_method = user_text, guess.language, guess.method
            m.stt_language = transcript.language
            await self.send.json(
                {
                    "type": "transcript",
                    "final": True,
                    "turn_id": m.turn_id,
                    "text": user_text,
                    "language": guess.language,
                    "language_method": guess.method,
                    "stt_language": transcript.language,
                }
            )
            self._set_busy(True)
            await self.send.json({"type": "state", "state": "thinking", "turn_id": m.turn_id})

            queue: asyncio.Queue[str | None] = asyncio.Queue()
            reply: list[str] = []
            tool_ms = 0.0

            async def produce() -> None:
                nonlocal tool_ms
                chunker = SentenceChunker(self.settings.first_chunk_min_chars)
                try:
                    async with contextlib.aclosing(
                        self.agent.run_turn(user_text, guess.language)
                    ) as events:
                        async for ev in events:
                            if isinstance(ev, AgentText):
                                reply.append(ev.text)
                                await self.send.json(
                                    {"type": "agent_text", "turn_id": m.turn_id, "delta": ev.text}
                                )
                                for chunk in chunker.push(ev.text):
                                    await queue.put(chunk)
                            elif isinstance(ev, AgentToolStart):
                                st.tools_ran = True
                                m.tools.append(ev.name)
                                await self.send.json(
                                    {
                                        "type": "tool_call",
                                        "turn_id": m.turn_id,
                                        "id": ev.call_id,
                                        "name": ev.name,
                                        "arguments": ev.arguments,
                                    }
                                )
                            elif isinstance(ev, AgentToolEnd):
                                tool_ms += ev.result.duration_ms
                                await self.send.json(
                                    {
                                        "type": "tool_result",
                                        "turn_id": m.turn_id,
                                        "id": ev.result.call.id,
                                        "ok": ev.result.ok,
                                        "output": ev.result.output,
                                        "duration_ms": round(ev.result.duration_ms, 2),
                                    }
                                )
                            elif isinstance(ev, AgentLLMRound):
                                m.llm_rounds += 1
                                if ev.round == 0:
                                    m.llm_ttft_ms = round(ev.ttft_ms, 1) if ev.ttft_ms else None
                                    m.prompt_tokens, m.cached_tokens = (
                                        ev.prompt_tokens,
                                        ev.cached_tokens,
                                    )
                                m.llm_total_ms = round((m.llm_total_ms or 0) + ev.total_ms, 1)
                    for chunk in chunker.flush():
                        await queue.put(chunk)
                finally:
                    await queue.put(None)

            async def consume() -> None:
                seq = 0
                while (chunk := await queue.get()) is not None:
                    t0 = time.perf_counter()
                    audio = bytearray()
                    async for part in p.tts.synthesize(chunk, guess.language):
                        if seq == 0 and not audio:
                            m.tts_ttfb_ms = _ms(t0)
                        audio.extend(part)
                    if not audio:
                        continue
                    if not st.audio_sent:
                        m.ttfa_ms, m.first_chunk_chars = _ms(t_end), len(chunk)
                        await self.send.json(
                            {"type": "state", "state": "speaking", "turn_id": m.turn_id}
                        )
                    await self.send.audio(
                        {
                            "type": "audio",
                            "turn_id": m.turn_id,
                            "seq": seq,
                            "format": p.tts.audio_format,
                            "text": chunk,
                            "bytes": len(audio),
                        },
                        bytes(audio),
                    )
                    st.audio_sent = True
                    seq += 1

            async with asyncio.TaskGroup() as tg:
                tg.create_task(produce())
                tg.create_task(consume())
            m.reply = "".join(reply).strip()
            m.tool_ms = round(tool_ms, 2) if m.tools else None
        except asyncio.CancelledError:
            m.interrupted = True
            raise
        except Exception as exc:
            m.error = f"{type(exc).__name__}: {exc}"
            await self.send.json({"type": "error", "turn_id": m.turn_id, "message": m.error})
        finally:
            m.turn_ms = _ms(t_end)
            # If audio went out, the client is still playing it; it reports playback_ended.
            if asyncio.current_task() is self._turn and not st.audio_sent:
                self._set_busy(False)
            if m.transcript:
                self.completed_turns.append(m)
                if self.metrics is not None:
                    self.metrics.save(m)
                with contextlib.suppress(Exception):
                    await self.send.json(
                        {"type": "turn_end", "turn_id": m.turn_id, "metrics": m.as_dict()}
                    )
