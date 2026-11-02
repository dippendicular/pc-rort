from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str
    redis_url: str
    s3_endpoint_url: str
    s3_access_key: str
    s3_secret_key: str
    s3_bucket: str = "pcrort-private"
    jwt_secret: str
    csrf_secret: str
    secure_cookies: bool = True
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024
    max_zip_members: int = 100_000
    max_uncompressed_bytes: int = 20 * 1024 * 1024 * 1024


settings = Settings()
