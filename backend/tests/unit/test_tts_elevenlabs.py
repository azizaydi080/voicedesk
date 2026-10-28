import json
from collections.abc import AsyncIterator, Callable

import httpx
import pytest
from pydantic import SecretStr

from voicedesk.config import Settings
from voicedesk.providers.registry import build_tts
from voicedesk.providers.tts import ElevenLabsTTS

VOICES = {"ar": "voice-ar", "fr": "voice-fr", "en": "voice-en"}


def _tts(
    handler: Callable[[httpx.Request], httpx.Response], model: str = "eleven_flash_v2_5"
) -> ElevenLabsTTS:
    client = httpx.AsyncClient(
        base_url="http://el/v1",
        headers={"xi-api-key": "test-key"},
        transport=httpx.MockTransport(handler),
    )
    return ElevenLabsTTS("test-key", VOICES, model, client=client)  # type: ignore[arg-type]


async def _collect(tts: ElevenLabsTTS, text: str, language: str = "fr") -> list[bytes]:
    return [c async for c in tts.synthesize(text, language)]  # type: ignore[arg-type]


async def test_request_shape_and_streamed_chunks() -> None:
    seen: dict = {}

    async def body() -> AsyncIterator[bytes]:
        for part in (b"ID3", b"\xff\xfb", b"", b"\x90\x00"):
            yield part

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, content=body(), headers={"content-type": "audio/mpeg"})

    chunks = await _collect(_tts(handler), "Bonjour, le cabinet est ouvert.")
    req = seen["request"]
    assert req.method == "POST"
    assert req.url.path == "/v1/text-to-speech/voice-fr/stream"
    assert req.url.params["output_format"] == "mp3_24000_48"
    assert "optimize_streaming_latency" not in req.url.params
    assert req.headers["xi-api-key"] == "test-key"
    assert json.loads(req.content) == {
        "text": "Bonjour, le cabinet est ouvert.",
        "model_id": "eleven_flash_v2_5",
        "language_code": "fr",
    }
    assert chunks == [b"ID3", b"\xff\xfb", b"\x90\x00"]  # streamed as received, no empties


@pytest.mark.parametrize("language", ["ar", "fr", "en"])
async def test_language_picks_voice_and_language_code(language: str) -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["body"] = request.url.path, json.loads(request.content)
        return httpx.Response(200, content=b"mp3")

    assert await _collect(_tts(handler), "x", language) == [b"mp3"]
    assert seen["path"] == f"/v1/text-to-speech/voice-{language}/stream"
    assert seen["body"]["language_code"] == language


async def test_multilingual_v2_omits_language_code() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"mp3")

    await _collect(_tts(handler, "eleven_multilingual_v2"), "مرحبا", "ar")
    assert seen["body"] == {"text": "مرحبا", "model_id": "eleven_multilingual_v2"}


@pytest.mark.parametrize(
    ("status", "detail"),
    [(401, "invalid_api_key"), (429, "concurrent_limit_exceeded"), (500, "server_error")],
)
async def test_http_errors_raise_with_status_and_detail(status: int, detail: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": {"status": detail}})

    with pytest.raises(RuntimeError, match=rf"ElevenLabs HTTP {status}: .*{detail}"):
        await _collect(_tts(handler), "x")


async def test_network_error_propagates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    with pytest.raises(httpx.ConnectError):
        await _collect(_tts(handler), "x")


def test_registry_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOICEDESK_ELEVENLABS_API_KEY", raising=False)
    s = Settings(tts="elevenlabs", _env_file=None)  # type: ignore[call-arg]
    with pytest.raises(RuntimeError, match="VOICEDESK_ELEVENLABS_API_KEY"):
        build_tts(s)


def test_registry_maps_voices_with_per_language_overrides() -> None:
    s = Settings(
        tts="elevenlabs",
        elevenlabs_api_key=SecretStr("k"),
        elevenlabs_voice_id="shared",
        elevenlabs_voice_id_ar="arabic",
        _env_file=None,  # type: ignore[call-arg]
    )
    tts = build_tts(s)
    assert isinstance(tts, ElevenLabsTTS)
    assert tts.voices == {"ar": "arabic", "fr": "shared", "en": "shared"}
    assert tts.model == "eleven_flash_v2_5"


def test_edge_stays_the_default() -> None:
    assert Settings(_env_file=None).tts == "edge"  # type: ignore[call-arg]
