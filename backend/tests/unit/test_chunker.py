from voicedesk.agent.chunker import SentenceChunker


def feed(chunker: SentenceChunker, text: str) -> list[str]:
    out: list[str] = []
    for ch in text:  # worst case: one character per token
        out += chunker.push(ch)
    return out + chunker.flush()


def test_first_chunk_cuts_at_first_clause() -> None:
    chunks = feed(
        SentenceChunker(first_min_chars=12),
        "Sure, tomorrow at ten works. Shall I book it for you now?",
    )
    assert chunks[0] == "Sure, tomorrow at ten works."[: len(chunks[0])]
    assert chunks[0].endswith((",", "."))
    assert " ".join(chunks) == "Sure, tomorrow at ten works. Shall I book it for you now?"


def test_short_first_clause_is_not_emitted_alone() -> None:
    chunks = feed(
        SentenceChunker(first_min_chars=12), "Yes, it is booked for tomorrow. See you then."
    )
    assert chunks[0] == "Yes, it is booked for tomorrow."


def test_arabic_punctuation() -> None:
    chunks = feed(
        SentenceChunker(first_min_chars=8), "نعم، الموعد غدا الساعة العاشرة. هل تريد شيئا آخر؟"
    )
    assert chunks[0] == "نعم، الموعد غدا الساعة العاشرة."
    assert chunks[-1].endswith("؟")


def test_later_chunks_wait_for_sentence_end() -> None:
    c = SentenceChunker(first_min_chars=5, min_chars=40)
    assert c.push("Hello there. ") == ["Hello there."]
    assert c.push("Short one. ") == []
    assert c.push("And now this is long enough to be spoken. ") == [
        "Short one. And now this is long enough to be spoken."
    ]
