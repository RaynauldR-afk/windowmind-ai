from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"
    window_size: int = Field(default=1200, ge=200)
    window_overlap: int = Field(default=200, ge=0)
    retrieval_k: int = Field(default=4, ge=1, le=12)
    max_context_chars: int = Field(default=9000, ge=1000)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def validate_window_configuration(self) -> None:
        if self.window_overlap >= self.window_size:
            raise ValueError("WINDOW_OVERLAP must be smaller than WINDOW_SIZE")

@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_window_configuration()
    return settings
