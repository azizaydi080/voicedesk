import io
import wave

import httpx

from voicedesk.providers.stt_openai_compat import OpenAICompatSTT
from tests.conftest import tone


async def test_posts_wav_and_maps_language() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.content
        return httpx.Response(200, json={"text": " مرحبا ", "language": "Arabic"})

    client = httpx.AsyncClient(base_url="http://stt/v1", transport=httpx.MockTransport(handler))
    stt = OpenAICompatSTT("http://stt/v1", "whisper-large-v3-turbo", None, client=client)
    stream = await stt.open_stream()
    await stream.send(tone(500))
    tr = await stream.finish()
    assert (tr.text, tr.language) == ("مرحبا", "ar")
    assert seen["path"] == "/v1/audio/transcriptions"
    assert b"RIFF" in seen["body"] and b"whisper-large-v3-turbo" in seen["body"]
    assert abs(tr.audio_seconds - 0.5) < 0.01


async def test_too_short_audio_skips_the_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    client = httpx.AsyncClient(base_url="http://stt", transport=httpx.MockTransport(handler))
    stt = OpenAICompatSTT("http://stt", "m", None, client=client)
    assert (await stt.transcribe(tone(100))).text == ""


def test_wav_header_is_16k_mono() -> None:
    from voicedesk.providers.stt_openai_compat import pcm16_to_wav

    with wave.open(io.BytesIO(pcm16_to_wav(tone(100))), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
