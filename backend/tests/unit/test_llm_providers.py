import json

import httpx
import pytest

from voicedesk.domain.tools import TOOL_SPECS
from voicedesk.providers.base import LLMDone, Message, TextDelta, ToolCall, ToolCallsReady
from voicedesk.providers.llm_llamacpp import ToolCallStreamParser, render_chatml
from voicedesk.providers.llm_openai_compat import OpenAICompatLLM

MODEL_OUTPUT = (
    'Let me check.\n<tool_call>\n{"name": "find_slots", "arguments": '
    '{"specialty": "dentist", "date": "tomorrow"}}\n</tool_call>'
)


@pytest.mark.parametrize("step", [1, 2, 3, 7, len(MODEL_OUTPUT)])
def test_tool_call_parser_never_leaks_markup(step: int) -> None:
    parser = ToolCallStreamParser()
    spoken = "".join(
        parser.feed(MODEL_OUTPUT[i : i + step]) for i in range(0, len(MODEL_OUTPUT), step)
    )
    tail, calls = parser.finish()
    assert "<" not in spoken + tail
    assert (spoken + tail).strip() == "Let me check."
    assert [(c.name, c.arguments) for c in calls] == [
        ("find_slots", {"specialty": "dentist", "date": "tomorrow"})
    ]


def test_parser_passes_plain_text_and_less_than_signs() -> None:
    parser = ToolCallStreamParser()
    out = parser.feed("Prices are < 100 dinars.")
    tail, calls = parser.finish()
    assert out + tail == "Prices are < 100 dinars." and calls == []


def test_parser_ignores_malformed_json() -> None:
    parser = ToolCallStreamParser()
    parser.feed("<tool_call>{not json}</tool_call>")
    assert parser.finish()[1] == []


def test_render_chatml_groups_tool_responses_and_keeps_prefix_stable() -> None:
    history = [
        Message("system", "SYS"),
        Message("user", "hi"),
        Message(
            "assistant",
            "",
            tool_calls=[
                ToolCall("a", "get_appointment", {"booking_ref": "1"}),
                ToolCall("b", "clinic_info", {"question": "x"}),
            ],
        ),
        Message("tool", '{"ok": 1}', tool_call_id="a"),
        Message("tool", '{"ok": 2}', tool_call_id="b"),
    ]
    prompt = render_chatml(history, TOOL_SPECS)
    assert prompt.count("<tool_response>") == 2
    assert prompt.count("<|im_start|>user") == 2  # both responses in one user block
    assert prompt.endswith("<|im_start|>assistant\n")
    longer = render_chatml(
        [*history, Message("assistant", "ok"), Message("user", "next")], TOOL_SPECS
    )
    assert longer.startswith(prompt.removesuffix("<|im_start|>assistant\n"))


def _sse(*chunks: dict) -> bytes:
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


async def test_openai_compat_streams_text_and_assembles_tool_calls() -> None:
    body = _sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "get_", "arguments": '{"booking'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "appointment", "arguments": '_ref": "1002"}'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 900,
                "completion_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 800},
            },
        },
    )
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(base_url="http://llm", transport=httpx.MockTransport(handler))
    llm = OpenAICompatLLM("http://llm", "m", client=client)
    events = [e async for e in llm.stream([Message("user", "where is 1002")], TOOL_SPECS)]
    assert seen["stream"] is True and len(seen["tools"]) == len(TOOL_SPECS)
    assert events[0] == ToolCallsReady([ToolCall("c1", "get_appointment", {"booking_ref": "1002"})])
    assert events[1] == LLMDone(900, 12, 800)


async def test_openai_compat_text_and_http_errors() -> None:
    ok = _sse(
        {"choices": [{"delta": {"content": "Hel"}}]}, {"choices": [{"delta": {"content": "lo"}}]}
    )
    client = httpx.AsyncClient(
        base_url="http://llm",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=ok)),
    )
    texts = [
        e.text
        async for e in OpenAICompatLLM("http://llm", "m", client=client).stream([], [])
        if isinstance(e, TextDelta)
    ]
    assert "".join(texts) == "Hello"

    bad = httpx.AsyncClient(
        base_url="http://llm",
        transport=httpx.MockTransport(lambda r: httpx.Response(429, content=b"rate limited")),
    )
    with pytest.raises(RuntimeError, match="429"):
        _ = [e async for e in OpenAICompatLLM("http://llm", "m", client=bad).stream([], [])]
