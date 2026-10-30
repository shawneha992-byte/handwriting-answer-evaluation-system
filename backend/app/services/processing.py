"""Background pipeline for one answer sheet: LLM reads it, code scores it, result is saved to MySQL."""
import time
from pathlib import Path

from app.core.config import settings
from app.core.database import SessionLocal
from app.models import Answer, Submission
from app.services import llm_service, scoring

MIME = {".pdf": "application/pdf", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def process_submission(sub_id: int) -> None:
    db = SessionLocal()
    try:
        sub = db.get(Submission, sub_id)
        sub.status = "reading"
        db.commit()
        exam = sub.exam
        questions = exam.questions
        try:
            path = Path(sub.stored_path)
            data = path.read_bytes()
            read = llm_service.parse_sheet(
                data, MIME[path.suffix.lower()],
                [{"number": q.number, "type": q.type, "correct_answer": q.correct_answer} for q in questions],
            )
        except Exception as exc:  # LLM or file problem: report it, never guess
            sub.status = "needs_review"
            sub.review_reason = f"The AI could not process this sheet: {exc}"
            sub.review_detail = "Whole sheet"
            db.commit()
            return

        sub.roll_no = (str(read.get("roll_no")).strip() if read.get("roll_no") else None) or None
        sub.student_name = read.get("name")
        reasons, unreadable = [], []

        if not sub.roll_no or (read.get("roll_confidence") or 0) < settings.MIN_CONFIDENCE:
            reasons.append("Roll number unreadable")
            unreadable.append("Roll number")
        elif db.query(Submission).filter(Submission.exam_id == exam.id, Submission.roll_no == sub.roll_no,
                                         Submission.id != sub.id, Submission.status.in_(["done", "needs_review"])).first():
            reasons.append(f"Duplicate roll number {sub.roll_no}")
            unreadable.append("Roll number")

        found = {a.get("number"): a for a in read.get("answers", [])}
        total = 0.0
        for q in questions:
            a = found.get(q.number, {})
            given = a.get("answer") if q.number in found else None
            result, marks = scoring.score_answer(q, given, a.get("confidence"), exam.negative_marking)
            if result == "unreadable":
                unreadable.append(f"Q{q.number}")
            total += marks
            db.add(Answer(submission_id=sub.id, question_number=q.number, student_answer="" if given is None else str(given),
                          correct_answer=q.correct_answer, result=result, marks=marks, confidence=a.get("confidence")))
        sub.total_score = max(0.0, total)  # score never goes below zero
        if any(u.startswith("Q") for u in unreadable):
            reasons.append("Some answers unreadable")
        if reasons:
            sub.status, sub.review_reason = "needs_review", "; ".join(reasons)
            sub.review_detail = ", ".join(unreadable)
        else:
            sub.status = "done"
        db.commit()
    finally:
        db.close()


def run_batch(ids: list[int]) -> None:
    for i, sub_id in enumerate(ids):
        process_submission(sub_id)
        if settings.LLM_PROVIDER == "gemini" and i < len(ids) - 1:
            time.sleep(settings.GEMINI_DELAY_SECONDS)  # stay inside the free-tier rate limit
