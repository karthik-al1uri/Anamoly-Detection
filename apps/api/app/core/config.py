from dataclasses import dataclass
import os


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Cold Start Detector API")
    api_prefix: str = "/api"
    mongo_url: str = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    mongo_db: str = os.getenv("MONGO_DB", "cold_start_detector")
    databricks_host: str = os.getenv("DATABRICKS_HOST", "")
    delta_table_name: str = os.getenv("DELTA_TABLE_NAME", "anomaly_events")
    cors_origins: tuple[str, ...] = _parse_csv(os.getenv("CORS_ORIGINS", "http://localhost:3000"))


settings = Settings()
