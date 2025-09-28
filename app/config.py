import os
from typing import Optional

class Settings:
    # Database settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://mlops_user:mlops_pass@postgres:5432/mlops_db"
    )
    
    # Model settings
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "distilbert")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "latest")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/app/models")
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    
    # Redis settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Performance
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))

settings = Settings()