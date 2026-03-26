from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    agent_mode: str = "ollama"
    openai_api_key: str

    #.env laden
    class Config:
        env_file = "../.env"

def get_settings():
    return Settings()
