from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotClientSettings(BaseSettings):
    """Runtime settings for the LiteIM BotClient connection."""

    liteim_host: str = Field(default="127.0.0.1", validation_alias="LITEIM_HOST")
    liteim_port: int = Field(default=9000, validation_alias="LITEIM_PORT", ge=1, le=65535)

    bot_username: str = Field(default="persona_agent_bot", validation_alias="BOT_USERNAME")
    bot_password: str = Field(default="change_me", validation_alias="BOT_PASSWORD")
    bot_nickname: str = Field(default="PersonaAgent", validation_alias="BOT_NICKNAME")
    bot_state_path: str = Field(
        default="data/bot_state/state.json",
        validation_alias="BOT_STATE_PATH",
    )
    offline_message_limit: int = Field(
        default=100,
        validation_alias="BOT_OFFLINE_MESSAGE_LIMIT",
        ge=1,
        le=100,
    )
    allowed_user_ids: str = Field(default="", validation_alias="BOT_ALLOWED_USER_IDS")
    allowed_usernames: str = Field(default="", validation_alias="BOT_ALLOWED_USERNAMES")
    auto_accept_friend_requests: bool = Field(
        default=True,
        validation_alias="BOT_AUTO_ACCEPT_FRIEND_REQUESTS",
    )
    reject_non_allowlisted_friend_requests: bool = Field(
        default=True,
        validation_alias="BOT_REJECT_NON_ALLOWLISTED_FRIEND_REQUESTS",
    )
    echo_mode: bool = Field(default=True, validation_alias="ECHO_MODE")
    agent_service_url: str = Field(
        default="http://127.0.0.1:8088",
        validation_alias="AGENT_SERVICE_URL",
    )
    agent_request_timeout_seconds: float = Field(
        default=5.0,
        validation_alias="AGENT_REQUEST_TIMEOUT_SECONDS",
        gt=0.0,
    )

    request_timeout_seconds: float = Field(
        default=5.0,
        validation_alias="BOT_REQUEST_TIMEOUT_SECONDS",
        gt=0.0,
    )
    heartbeat_interval_seconds: float = Field(
        default=30.0,
        validation_alias="BOT_HEARTBEAT_INTERVAL_SECONDS",
        gt=0.0,
    )
    reconnect_initial_delay_seconds: float = Field(
        default=0.2,
        validation_alias="BOT_RECONNECT_INITIAL_DELAY_SECONDS",
        ge=0.0,
    )
    reconnect_max_delay_seconds: float = Field(
        default=5.0,
        validation_alias="BOT_RECONNECT_MAX_DELAY_SECONDS",
        ge=0.0,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )
