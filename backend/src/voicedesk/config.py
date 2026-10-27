"""Runtime configuration, loaded from environment variables (prefix ``VOICEDESK_``) or ``.env``."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

STTName = Literal["whisper", "deepgram", "fake"]
LLMName = Literal["llamacpp", "openai_compat", "scripted"]
TTSName = Literal["edge", "elevenlabs", "fake"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOICEDESK_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- provider selection -------------------------------------------------
    stt: STTName = "whisper"
    llm: LLMName = "llamacpp"
    tts: TTSName = "edge"

    # --- STT ----------------------------------------------------------------
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 1
    deepgram_api_key: SecretStr | None = None
    deepgram_model: str = "nova-3"
    deepgram_language: str = "multi"

    # --- LLM ----------------------------------------------------------------
    llamacpp_model_path: Path = Path("../.local/models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    llamacpp_threads: int | None = None
    llamacpp_ctx: int = 4096
    openai_base_url: str = "https://api.groq.com/openai/v1"
    openai_api_key: SecretStr | None = None
    openai_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 220
    max_tool_rounds: int = 3

    # --- TTS ----------------------------------------------------------------
    edge_voice_ar: str = "ar-SA-ZariyahNeural"
    edge_voice_fr: str = "fr-FR-DeniseNeural"
    edge_voice_en: str = "en-US-AriaNeural"
    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model: str = "eleven_flash_v2_5"

    # --- turn taking --------------------------------------------------------
    vad_threshold_db: float = Field(default=12.0, description="dB above noise floor = speech")
    vad_start_ms: int = 60
    vad_end_silence_ms: int = 550
    barge_in_min_ms: int = 180
    first_chunk_min_chars: int = 12

    # --- domain / storage ---------------------------------------------------
    clinic_today: date | None = None
    metrics_path: Path = Path("data/metrics.sqlite")
    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
