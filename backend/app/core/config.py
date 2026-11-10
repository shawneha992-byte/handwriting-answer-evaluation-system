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
    OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
    MIN_PAGE_TEXT_CHARS = int(os.getenv("MIN_PAGE_TEXT_CHARS", "40"))
    OCR_ZOOM = float(os.getenv("OCR_ZOOM", "3"))

    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
    FRONTEND_DIR = BASE_DIR.parent / "frontend"
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
    PROCESSING_WORKERS = int(os.getenv("PROCESSING_WORKERS", "2"))

    MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.6"))
    NUMERIC_TOLERANCE = float(os.getenv("NUMERIC_TOLERANCE", "0.01"))

    # Sheets that never finished (server stopped mid-batch)
    RECOVER_ON_START = os.getenv("RECOVER_ON_START", "true").strip().lower() not in ("0", "false", "no", "off")
    STALE_MINUTES = int(os.getenv("STALE_MINUTES", "30"))  # a sheet "reading" for longer than this is treated as stuck

    # Optional second reading of each sheet at a lower resolution; any answer the two readings disagree on goes to review
    SECOND_READ = os.getenv("SECOND_READ", "false").strip().lower() in ("1", "true", "yes", "on")
    SECOND_READ_MAX_DIM = int(os.getenv("SECOND_READ_MAX_DIM", "1200"))

    # Retrieval-augmented grading of descriptive answers
    EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama").strip().lower()  # ollama | hash (hash is for tests)
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
    GRADER_MODEL = os.getenv("OLLAMA_GRADER_MODEL", "llama3.1:8b")  # a text model; the vision model only reads handwriting
    RAG_CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "900"))
    RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))
    RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))  # best cosine below this = weak match; tune on your own data
    RAG_MARK_STEP = float(os.getenv("RAG_MARK_STEP", "0.25"))
    # nomic-embed-text returns 768 dimensions; PostgreSQL pgvector column uses this fixed dimension.
    RAG_EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "768"))

    # Login / registration (authentication layer)
    AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "true").strip().lower() not in ("0", "false", "no", "off")
    AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "")  # leave empty to auto-create backend/.auth_secret
    AUTH_TOKEN_HOURS = int(os.getenv("AUTH_TOKEN_HOURS", "12"))

    # Forgot password: the link in the email points to APP_BASE_URL. Without SMTP_HOST the link is printed in the server console.
    APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    RESET_TOKEN_MINUTES = int(os.getenv("RESET_TOKEN_MINUTES", "30"))
    SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM = os.getenv("SMTP_FROM", "") or os.getenv("SMTP_USER", "")
    SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "true").strip().lower() not in ("0", "false", "no", "off")


settings = Settings()
