"""Concurrent student-sheet pipeline with atomic queue claiming and review safeguards.

The configured vision model reads handwriting; scoring.py alone calculates marks. Each worker owns
its own SQLAlchemy session. Uncertain or malformed readings go to manual review.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.core.config import settings
from app.core.database import SessionLocal
from app.models import Answer, Submission
from app.services import llm_service, scoring

MIME = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _mark_for_review(sub, reason: str, detail: str = "Whole sheet") -> None:
    sub.status = "needs_review"
    sub.review_reason = str(reason)[:2000]
    sub.review_detail = str(detail)[:255]


def _confidence(value) -> float:
    """Convert provider confidence to a safe 0..1 float."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, value))


def _claim_submission(db, sub_id: int) -> bool:
    """Atomically claim only queued work; prevents two workers scoring one sheet."""
    changed = (
        db.query(Submission)
        .filter(Submission.id == sub_id, Submission.status == "queued")
        .update(
            {Submission.status: "reading"},
            synchronize_session=False,
        )
    )
    db.commit()
    return changed == 1


def process_submission(sub_id: int) -> None:
    """Process one queued sheet. Duplicate worker calls safely become no-ops."""
    db = SessionLocal()
    try:
        sub = db.get(Submission, sub_id)
        if sub is None or sub.status != "queued":
            return

        if not _claim_submission(db, sub_id):
            return

        # Refresh after the atomic claim so this session sees the committed state.
        db.expire_all()
        sub = db.get(Submission, sub_id)
        if sub is None:
            return

        sub.review_reason = None
        sub.review_detail = None

        exam = sub.exam
        questions = list(exam.questions)
        if not questions:
            _mark_for_review(sub, "This exam has no saved questions.")
            db.commit()
            return

        path = Path(sub.stored_path)
        suffix = path.suffix.lower()
        if suffix not in MIME or not path.is_file():
            _mark_for_review(
                sub,
                f"Unsupported or missing answer-sheet file: {suffix or 'unknown'}",
            )
            db.commit()
            return

        # Include question wording and options as context. Never send the answer key:
        # the model's job is transcription, not deciding whether an answer is correct.
        question_context = [
            {
                "number": q.number,
                "type": q.type,
                "text": q.text or "",
            }
            for q in questions
        ]

        try:
            read = llm_service.parse_sheet(
                path.read_bytes(), MIME[suffix], question_context
            )
            if not isinstance(read, dict):
                raise ValueError("The reader returned an invalid response.")
            raw_answers = read.get("answers")
            if not isinstance(raw_answers, list):
                raise ValueError("The reader did not return an answers list.")
        except Exception as exc:
            _mark_for_review(sub, f"The AI could not process this sheet: {exc}")
            db.commit()
            return

        roll = read.get("roll_no")
        sub.roll_no = (
            str(roll).strip()[:50]
            if roll is not None and str(roll).strip()
            else None
        )
        name = read.get("name")
        sub.student_name = str(name).strip()[:120] if name else None

        reasons, review_items = [], []
        # Some vision models omit confidence despite returning a legible value.
        # Use a conservative default in that case; never invent a missing ID.
        roll_conf_raw = read.get("roll_confidence")
        roll_conf = _confidence(roll_conf_raw if roll_conf_raw is not None else (0.75 if sub.roll_no else 0.0))
        if not sub.roll_no or roll_conf < settings.MIN_CONFIDENCE:
            reasons.append("Roll number unreadable")
            review_items.append("Roll number")
        else:
            duplicate = (
                db.query(Submission)
                .filter(
                    Submission.exam_id == exam.id,
                    Submission.roll_no == sub.roll_no,
                    Submission.id != sub.id,
                    Submission.status.in_(["done", "needs_review"]),
                )
                .first()
            )
            if duplicate:
                reasons.append(f"Duplicate roll number {sub.roll_no}")
                review_items.append("Roll number")

        # Keep the first valid occurrence. Conflicting duplicate question numbers
        # are treated as unreadable instead of silently accepting the last value.
        expected = {q.number for q in questions}
        found, duplicate_numbers = {}, set()
        for item in raw_answers:
            if not isinstance(item, dict):
                continue
            try:
                number = int(item.get("number"))
            except (TypeError, ValueError):
                continue
            if number not in expected:
                continue
            if number in found:
                duplicate_numbers.add(number)
            else:
                found[number] = item

        db.query(Answer).filter(
            Answer.submission_id == sub.id
        ).delete(synchronize_session=False)

        total = 0.0
        answer_rows = []
        for q in questions:
            item = found.get(q.number)
            if q.number in duplicate_numbers:
                given, confidence = None, 0.0
            elif item is None:
                given, confidence = None, 0.0
            else:
                given = item.get("answer")
                raw_confidence = item.get("confidence")
                confidence = _confidence(
                    raw_confidence if raw_confidence is not None
                    else (1.0 if given is not None and str(given).strip() == "" else 0.75 if given is not None else 0.0)
                )

            result, marks = scoring.score_answer(
                q, given, confidence, exam.negative_marking
            )
            if result == "unreadable":
                review_items.append(f"Q{q.number}")
            total += marks
            answer_rows.append(
                Answer(
                    submission_id=sub.id,
                    question_number=q.number,
                    student_answer="" if given is None else str(given).strip()[:255],
                    correct_answer=q.correct_answer,
                    result=result,
                    marks=marks,
                    confidence=confidence,
                )
            )

        db.add_all(answer_rows)
        sub.total_score = max(0.0, total)

        if review_items:
            if any(item.startswith("Q") for item in review_items):
                reasons.append("Some answers unreadable")
            _mark_for_review(sub, "; ".join(dict.fromkeys(reasons)), ", ".join(dict.fromkeys(review_items)))
        else:
            sub.status = "done"
            sub.review_reason = None
            sub.review_detail = None

        db.commit()

    except Exception as exc:
        db.rollback()
        # Preserve the per-sheet failure; one bad upload must not stop the batch.
        try:
            sub = db.get(Submission, sub_id)
            if sub is not None and sub.status != "done":
                _mark_for_review(sub, f"Unexpected processing error: {exc}")
                db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()


def run_batch(ids: list[int]) -> None:
    """Run local vision jobs sequentially; retain bounded concurrency for cloud providers.

    Ollama is a local model server and multiple image-generation/vision requests can
    compete for the same RAM/VRAM. Sequential processing avoids that resource spike.
    Each submission is isolated, so a failure never prevents later sheets from running.
    """
    if not ids:
        return

    if str(settings.LLM_PROVIDER).lower() == "ollama":
        for sub_id in ids:
            try:
                process_submission(sub_id)
            except Exception:
                # process_submission normally records its own error; continue the batch
                # even if an unexpected exception escapes.
                continue
        return

    try:
        workers = int(os.getenv("PROCESSING_WORKERS", "2"))
    except (TypeError, ValueError):
        workers = 2
    workers = max(1, min(workers, 5, len(ids)))

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="sheet-worker"
    ) as pool:
        futures = [pool.submit(process_submission, sub_id) for sub_id in ids]
        for future in as_completed(futures):
            future.result()
