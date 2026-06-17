from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_pretty: bool = False
    ocr_max_dimension: int = 2048
    max_upload_bytes: int = 10 * 1024 * 1024


settings = Settings()
