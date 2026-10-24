from datetime import time

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_env: str = "dev"
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/voice_agent"
    )
    internal_api_secret: str = "change-me"
    internal_api_url: str = "http://localhost:8000"
    public_base_url: str = "http://localhost:8000"

    # Call rules
    call_timezone: str = "Europe/Warsaw"
    call_hours_start: time = time(9, 0)
    call_hours_end: time = time(20, 0)
    retry_after_minutes: int = 15
    max_call_attempts: int = 2
    participant_join_timeout: float = 60.0

    # LiveKit
    livekit_url: str = "wss://your-project.livekit.cloud"
    livekit_api_key: str | None = None
    livekit_api_secret: str | None = None
    sip_outbound_trunk_id: str | None = None
    # Empty for automatic dispatch (Playground/dev), set name for explicit dispatch
    agent_dispatch_name: str = ""

    # Models & Voice Pipeline
    llm_provider: str = "google"
    llm_model: str = "gemini-3.5-flash-lite"
    llm_fallback_model: str | None = "gemini-3.5-flash"
    google_api_key: str | None = None
    openai_api_key: str | None = None

    # Post-call Extraction
    extraction_model: str = "gemini-3.5-flash"
    extraction_timeout: float = 15.0

    stt_provider: str = "deepgram"
    stt_model: str | None = None
    # Optional STT language override; if None, language from script.yaml is used
    stt_language: str | None = None
    stt_keyterms: str | None = None
    deepgram_api_key: str | None = None

    tts_provider: str = "elevenlabs"
    tts_model: str | None = None
    tts_voice_id: str | None = None
    elevenlabs_api_key: str | None = None

    # Turn detection & Endpointing
    # "vad" (local Silero, free) or "turn_detector" (cloud inference)
    turn_detection_mode: str = "vad"
    # "vad" (local Silero, free) or "adaptive" (cloud inference)
    interruption_mode: str = "vad"
    # "fixed" or "dynamic"
    endpointing_mode: str = "fixed"
    endpointing_min_delay: float = 1.0
    endpointing_max_delay: float = 3.0
    preemptive_generation: bool = True
    end_call_grace_period: float = 1.2
    end_call_noise_timeout_cap: float = 3.0

    # Delivery
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    google_sheets_id: str | None = None
    google_service_account_json: str | None = None


settings = Settings()
