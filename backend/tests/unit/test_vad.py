from tests.conftest import silence, tone
from voicedesk.audio.vad import FRAME_MS, EnergyVAD, VadEvent


def events(vad: EnergyVAD, pcm: bytes) -> list[tuple[int, VadEvent]]:
    return [(i * FRAME_MS, ev) for i, (_, ev) in enumerate(vad.process(pcm)) if ev]


def test_detects_start_and_end_with_configured_timing() -> None:
    vad = EnergyVAD(start_ms=60, end_silence_ms=500)
    got = events(vad, silence(400) + tone(800) + silence(700))
    assert [e for _, e in got] == [VadEvent.SPEECH_START, VadEvent.SPEECH_END]
    start_ms, end_ms = got[0][0], got[1][0]
    assert 400 <= start_ms <= 400 + 60 + FRAME_MS
    assert 1200 + 500 - FRAME_MS <= end_ms <= 1200 + 500 + FRAME_MS


def test_short_click_is_not_speech() -> None:
    vad = EnergyVAD(start_ms=60)
    assert events(vad, silence(300) + tone(40) + silence(300)) == []


def test_brief_pause_does_not_end_utterance() -> None:
    vad = EnergyVAD(end_silence_ms=500)
    got = events(vad, silence(200) + tone(400) + silence(300) + tone(400) + silence(800))
    assert [e for _, e in got] == [VadEvent.SPEECH_START, VadEvent.SPEECH_END]


def test_handles_arbitrary_chunk_sizes() -> None:
    vad = EnergyVAD()
    pcm = silence(300) + tone(500) + silence(800)
    out = []
    for i in range(0, len(pcm), 777):  # not frame aligned
        out += [ev for _, ev in vad.process(pcm[i : i + 777]) if ev]
    assert out == [VadEvent.SPEECH_START, VadEvent.SPEECH_END]


def test_steady_background_is_calibrated_out() -> None:
    vad = EnergyVAD()
    noise = tone(3000, freq=50, amp=0.01)  # ~-43 dBFS hum from the moment the mic opens
    assert events(vad, noise) == []
    assert vad.floor_db > -50


def test_noise_that_starts_later_is_absorbed() -> None:
    vad = EnergyVAD(end_silence_ms=500)
    got = events(vad, silence(400) + tone(6000, freq=50, amp=0.01))
    assert [e for _, e in got] == [VadEvent.SPEECH_START, VadEvent.SPEECH_END]
    assert not vad.speaking


def test_speech_is_detected_over_background_noise() -> None:
    vad = EnergyVAD()
    noise = tone(1000, freq=50, amp=0.01)
    got = events(vad, noise + tone(600, amp=0.3) + noise)
    assert [e for _, e in got] == [VadEvent.SPEECH_START, VadEvent.SPEECH_END]
