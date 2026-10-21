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
    public_base_url: str = "http://localhost:8000"

    # Call rules
    call_timezone: str = "Europe/Warsaw"
    call_hours_start: time = time(9, 0)
    call_hours_end: time = time(20, 0)
    retry_after_minutes: int = 15
    max_call_attempts: int = 2

    # LiveKit
    livekit_url: str = "wss://your-project.livekit.cloud"
    livekit_api_key: str | None = None
    livekit_api_secret: str | None = None
    livekit_sip_outbound_trunk_id: str | None = None
    agent_name: str = "callback-agent"

    # Models
    openai_api_key: str | None = None
    llm_model: str | None = None
    deepgram_api_key: str | None = None
    stt_language: str = "pl"
    elevenlabs_api_key: str | None = None
    tts_voice_id: str | None = None

    # Delivery
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    google_sheets_id: str | None = None
    google_service_account_json: str | None = None


settings = Settings()
