"""Builds providers from settings. The only place that knows concrete classes."""

from __future__ import annotations

from dataclasses import dataclass

from voicedesk.config import Settings
from voicedesk.providers.base import LLMProvider, STTProvider, TTSProvider


@dataclass(frozen=True, slots=True)
class Providers:
    stt: STTProvider
    llm: LLMProvider
    tts: TTSProvider


def _require(secret: object, var: str) -> str:
    value = secret.get_secret_value() if secret is not None else ""  # type: ignore[attr-defined]
    if not value:
        raise RuntimeError(f"{var} must be set for this provider (see .env.example).")
    return str(value)


def build_stt(s: Settings) -> STTProvider:
    match s.stt:
        case "whisper":
            from voicedesk.providers.stt_whisper import WhisperSTT

            return WhisperSTT(s.whisper_model, s.whisper_compute_type, s.whisper_beam_size)
        case "deepgram":
            from voicedesk.providers.stt_deepgram import DeepgramSTT

            key = _require(s.deepgram_api_key, "VOICEDESK_DEEPGRAM_API_KEY")
            return DeepgramSTT(key, s.deepgram_model, s.deepgram_language)
        case "fake":
            from voicedesk.providers.fakes import FakeSTT

            return FakeSTT()


def build_llm(s: Settings) -> LLMProvider:
    match s.llm:
        case "llamacpp":
            from voicedesk.providers.llm_llamacpp import LlamaCppLLM

            return LlamaCppLLM(
                s.llamacpp_model_path,
                s.llamacpp_ctx,
                s.llamacpp_threads,
                s.llm_temperature,
                s.llm_max_tokens,
            )
        case "openai_compat":
            from voicedesk.providers.llm_openai_compat import OpenAICompatLLM

            key = s.openai_api_key.get_secret_value() if s.openai_api_key else None
            return OpenAICompatLLM(
                s.openai_base_url, s.openai_model, key, s.llm_temperature, s.llm_max_tokens
            )
        case "scripted":
            from voicedesk.providers.fakes import ScriptedLLM

            return ScriptedLLM()


def build_tts(s: Settings) -> TTSProvider:
    match s.tts:
        case "edge":
            from voicedesk.providers.tts import EdgeTTS

            return EdgeTTS({"ar": s.edge_voice_ar, "fr": s.edge_voice_fr, "en": s.edge_voice_en})
        case "elevenlabs":
            from voicedesk.providers.tts import ElevenLabsTTS

            key = _require(s.elevenlabs_api_key, "VOICEDESK_ELEVENLABS_API_KEY")
            return ElevenLabsTTS(key, s.elevenlabs_voice_id, s.elevenlabs_model)
        case "fake":
            from voicedesk.providers.fakes import FakeTTS

            return FakeTTS()


def build_providers(s: Settings) -> Providers:
    return Providers(build_stt(s), build_llm(s), build_tts(s))
