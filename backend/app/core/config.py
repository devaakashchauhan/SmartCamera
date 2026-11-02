from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseModel):
    APP_NAME: str = "Smart Camera API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Storage & Database Paths
    STORAGE_DIR: str = str(BASE_DIR / "storage")
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/storage/smart_camera.db"
    
    # CORS Origins
    CORS_ORIGINS: list[str] = ["*"]

settings = Settings()
# Ensure storage folder always exists
Path(settings.STORAGE_DIR).mkdir(parents=True, exist_ok=True)
