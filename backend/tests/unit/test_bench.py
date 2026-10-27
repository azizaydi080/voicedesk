from voicedesk.bench import speech_end_offset_s, summarize_rows
from tests.conftest import silence, tone


def test_speech_end_ignores_trailing_silence() -> None:
    pcm = silence(300) + tone(1000) + silence(900)
    assert abs(speech_end_offset_s(pcm) - 1.3) <= 0.02


def test_summary_separates_throttled_turns() -> None:
    rows = [
        {"ttfa_ms": 1000.0, "llm_throttled_ms": 0.0, "llm_ttft_ms": 300.0},
        {"ttfa_ms": 9000.0, "llm_throttled_ms": 7000.0, "llm_ttft_ms": 7300.0},
        {"ttfa_ms": 1200.0, "llm_throttled_ms": 0.0, "endpoint_splits": 1},
    ]
    s = summarize_rows(rows)
    assert s["throttled_turns"] == 1
    assert s["unthrottled"]["ttfa_ms"]["n"] == 2
    assert s["unthrottled"]["ttfa_ms"]["p90"] <= 1200.0
    assert s["turns_split_by_endpointer"] == 1
