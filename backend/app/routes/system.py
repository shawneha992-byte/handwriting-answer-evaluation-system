import requests
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.access import optional_user
from app.core.config import settings
from app.core.database import get_db
from app.models import Exam, Submission, User

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/config")
def config():
    provider = str(settings.LLM_PROVIDER).lower()
    message = ""

    if provider == "ollama":
        model = settings.OLLAMA_MODEL
        try:
            response = requests.get(f"{settings.OLLAMA_URL.rstrip('/')}/api/tags", timeout=2)
            response.raise_for_status()
            installed = [m.get("name", "") for m in response.json().get("models", [])]
            ready = any(name == model or name.split(":")[0] == model.split(":")[0] for name in installed)
            if not ready:
                message = f"Ollama is running, but model '{model}' is not installed. Run: ollama pull {model}"
        except requests.RequestException as exc:
            ready = False
            message = f"Cannot reach Ollama at {settings.OLLAMA_URL}. Start Ollama and retry. ({exc})"
        except (ValueError, TypeError):
            ready = False
            message = "Ollama responded, but its model list could not be read."
    elif provider == "gemini":
        model = settings.GEMINI_MODEL
        ready = bool(settings.GEMINI_API_KEY)
        if not ready:
            message = "GEMINI_API_KEY is missing from backend/.env."
    elif provider == "mock":
        model, ready = "mock data", True
        message = "Mock mode returns sample data only; it does not read real handwriting."
    else:
        model, ready = provider, False
        message = f"Unsupported LLM_PROVIDER '{provider}'. Use ollama, gemini, or mock."

    return {
        "provider": provider,
        "model": model,
        "llm_ready": ready,
        "llm_message": message,
        "database": settings.DATABASE_URL.split(":")[0].split("+")[0],
    }


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    query = db.query(Exam)
    if user is not None:
        query = query.filter(Exam.owner_id == user.id)
    exams = query.order_by(Exam.id.desc()).all()
    ids = [e.id for e in exams]
    mine = Submission.exam_id.in_(ids) if ids else Submission.id < 0  # no exams -> match nothing
    done = db.query(Submission).filter(Submission.status == "done", mine).all()
    pcts = [round((s.total_score or 0) / s.exam.total_marks * 100) for s in done if s.exam.total_marks]
    recent = []
    for e in exams:
        busy = any(s.status in ("queued", "reading") for s in e.submissions)
        review = sum(s.status == "needs_review" for s in e.submissions)
        recent.append({"id": e.id, "title": e.title, "sheets": len(e.submissions), "needs_review": review,
                       "status": "Checking" if busy else ("Needs review" if review else "Done")})
    return {"exams": len(exams), "sheets_checked": len(done),
            "average": round(sum(pcts) / len(pcts)) if pcts else 0,
            "needs_review": db.query(Submission).filter(Submission.status == "needs_review", mine).count(), "recent": recent}
