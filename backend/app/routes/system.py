from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models import Exam, Submission

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/config")
def config():
    ready = settings.LLM_PROVIDER == "mock" or bool(settings.GEMINI_API_KEY)
    return {"provider": settings.LLM_PROVIDER, "model": settings.GEMINI_MODEL if settings.LLM_PROVIDER == "gemini" else "mock data",
            "llm_ready": ready, "database": settings.DATABASE_URL.split(":")[0].split("+")[0]}


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db)):
    exams = db.query(Exam).order_by(Exam.id.desc()).all()
    done = db.query(Submission).filter(Submission.status == "done").all()
    pcts = [round((s.total_score or 0) / s.exam.total_marks * 100) for s in done if s.exam.total_marks]
    recent = []
    for e in exams[:5]:
        busy = any(s.status in ("queued", "reading") for s in e.submissions)
        review = sum(s.status == "needs_review" for s in e.submissions)
        recent.append({"id": e.id, "title": e.title, "sheets": len(e.submissions), "needs_review": review,
                       "status": "Checking" if busy else ("Needs review" if review else "Done")})
    return {"exams": len(exams), "sheets_checked": len(done),
            "average": round(sum(pcts) / len(pcts)) if pcts else 0,
            "needs_review": db.query(Submission).filter(Submission.status == "needs_review").count(), "recent": recent}
