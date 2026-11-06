"""Central settings loaded from backend/.env."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class Settings:
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:password@localhost:5432/answer_eval",
    )

    # Supported providers: ollama, gemini, mock
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_DELAY_SECONDS = float(os.getenv("GEMINI_DELAY_SECONDS", "4"))

    # Keep both URL names: older modules use OLLAMA_URL, newer code uses BASE_URL.
    OLLAMA_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
    ).rstrip("/")
    OLLAMA_URL = os.getenv("OLLAMA_URL", OLLAMA_BASE_URL).rstrip("/")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:4b-instruct")
    OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
    OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))
    OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))
    OLLAMA_RETRY_DELAY_SECONDS = float(os.getenv("OLLAMA_RETRY_DELAY_SECONDS", "3"))
    OLLAMA_MAX_IMAGE_DIMENSION = int(os.getenv("OLLAMA_MAX_IMAGE_DIMENSION", "1800"))
    OLLAMA_JPEG_QUALITY = int(os.getenv("OLLAMA_JPEG_QUALITY", "88"))

    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
    MIN_PAGE_TEXT_CHARS = int(os.getenv("MIN_PAGE_TEXT_CHARS", "40"))
    OCR_ZOOM = float(os.getenv("OCR_ZOOM", "3"))

    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
    FRONTEND_DIR = BASE_DIR.parent / "frontend"
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))

    MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.6"))
    NUMERIC_TOLERANCE = float(os.getenv("NUMERIC_TOLERANCE", "0.01"))


settings = Settings()
