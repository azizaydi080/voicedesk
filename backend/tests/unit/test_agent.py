import asyncio
import contextlib

from tests.conftest import TODAY
from voicedesk.agent.agent import Agent, AgentLLMRound, AgentText, AgentToolEnd, AgentToolStart
from voicedesk.domain.clinic import ClinicStore
from voicedesk.domain.tools import ToolRunner
from voicedesk.providers.fakes import ScriptedLLM


def make_agent(delay: float = 0.0) -> Agent:
    store = ClinicStore(TODAY)
    return Agent(ScriptedLLM(token_delay_s=delay), ToolRunner(store), TODAY)


async def test_tool_loop_runs_tool_then_answers() -> None:
    agent = make_agent()
    events = [e async for e in agent.run_turn("Please cancel booking 1002", "en")]
    kinds = [type(e) for e in events]
    assert kinds[:3] == [AgentLLMRound, AgentToolStart, AgentToolEnd]
    end = next(e for e in events if isinstance(e, AgentToolEnd))
    assert end.result.ok and end.result.output["status"] == "cancelled"
    assert any(isinstance(e, AgentText) for e in events)
    roles = [m.role for m in agent.conversation]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert agent.tools.store.bookings["1002"].status == "cancelled"


async def test_language_note_is_part_of_user_message() -> None:
    agent = make_agent()
    _ = [e async for e in agent.run_turn("Bonjour", "fr")]
    assert "French" in agent.conversation[0].content


async def test_interrupted_turn_is_committed_as_partial() -> None:
    agent = make_agent(delay=0.02)

    async def consume() -> None:
        async with contextlib.aclosing(agent.run_turn("hello", "en")) as events:
            async for _ in events:
                pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.07)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert agent.history[-1].role == "assistant"
    assert agent.history[-1].content.endswith("[interrupted by caller]")
