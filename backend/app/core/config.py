"""Central settings, read from environment variables or backend/.env"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]  # the backend/ folder
load_dotenv(BASE_DIR / ".env")


class Settings:
    # Database (MySQL by default; sqlite:///./dev.db also works for quick tests)
    DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:password@localhost:3306/answer_eval")

    # LLM: "gemini" (free tier) or "mock" (fake data, no key needed, for trying the UI)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_DELAY_SECONDS = float(os.getenv("GEMINI_DELAY_SECONDS", "4"))  # pause between sheets (free-tier rate limit)

    # Files
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
    FRONTEND_DIR = BASE_DIR.parent / "frontend"
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))

    # Scoring
    MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.6"))  # below this an answer is treated as unreadable
    NUMERIC_TOLERANCE = float(os.getenv("NUMERIC_TOLERANCE", "0.01"))


settings = Settings()
