"""Concurrent student-sheet pipeline with atomic queue claiming and review safeguards.

The configured vision model reads handwriting; scoring.py alone calculates marks. Each worker owns
its own SQLAlchemy session. Uncertain or malformed readings go to manual review.
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from app.core.config import settings
from app.core.database import SessionLocal
from app.models import Answer, Submission
import json

from app.rag import grader as rag_grader
from app.rag.embeddings import get_embedder
from app.rag.retrieve import KnowledgeIndex
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


def _collect(raw_answers, expected: set[int]) -> tuple[dict, set]:
    """Keep the first valid entry per question number. Conflicting duplicates are reported, not silently accepted."""
    found, duplicates = {}, set()
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
            duplicates.add(number)
        else:
            found[number] = item
    return found, duplicates


def _value(item):
    """None = could not be read, "" = left blank, otherwise the text."""
    if not isinstance(item, dict) or item.get("answer") is None:
        return None
    return str(item["answer"]).strip()


def _same_reading(first, second, qtype: str) -> bool:
    a, b = _value(first), _value(second)
    if a is None or b is None:
        return a is None and b is None
    if str(qtype).lower() == "numeric":
        x, y = scoring.to_number(a), scoring.to_number(b)
        if x is not None and y is not None:
            return abs(x - y) <= settings.NUMERIC_TOLERANCE
        return a.casefold() == b.casefold()
    return scoring.norm_mcq(a) == scoring.norm_mcq(b)


def _roll_key(value) -> str:
    return re.sub(r"\s+", "", str(value or "")).upper()


def recover_stuck(db, exam_id: int | None = None) -> list[int]:
    """Sheets that never finished, for example because the server stopped in the middle of a batch.

    A sheet that has been 'reading' for longer than STALE_MINUTES is put back to 'queued'. Returns the ids of every
    queued sheet so the caller can run them. Running a sheet twice is safe: the claim in process_submission is atomic.
    """
    cutoff = datetime.utcnow() - timedelta(minutes=settings.STALE_MINUTES)
    stale = db.query(Submission).filter(Submission.status == "reading", Submission.updated_at < cutoff)
    queued = db.query(Submission.id).filter(Submission.status == "queued")
    if exam_id is not None:
        stale, queued = stale.filter(Submission.exam_id == exam_id), queued.filter(Submission.exam_id == exam_id)
    stale.update({Submission.status: "queued"}, synchronize_session=False)
    db.commit()
    return [row[0] for row in queued.order_by(Submission.id)]


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


def _grade_descriptive(q, answer_text: str, index, index_error: str | None) -> dict:
    """Retrieve evidence for one question and let the grader judge the transcribed answer."""
    payload = rag_grader.question_payload(q)
    if index_error is not None:
        return {"result": "unreadable", "marks": 0.0, "feedback": "", "citations": [], "retrieval_score": None,
                "needs_review": True, "review_reason": index_error}
    hits = index.search_many(rag_grader.retrieval_queries(payload), settings.RAG_TOP_K) if index else []
    return rag_grader.evaluate(payload, answer_text, hits)


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

        second, second_error = None, None
        if settings.SECOND_READ:
            # A model's own confidence is only its opinion. Reading the sheet again at a lower resolution and
            # comparing the two readings catches errors the confidence number does not.
            try:
                second = llm_service.parse_sheet(
                    path.read_bytes(), MIME[suffix], question_context, max_dim=settings.SECOND_READ_MAX_DIM
                )
                if not isinstance(second, dict) or not isinstance(second.get("answers"), list):
                    raise ValueError("no answers list")
            except Exception as exc:
                second, second_error = None, str(exc)

        roll = read.get("roll_no")
        sub.roll_no = (
            str(roll).strip()[:50]
            if roll is not None and str(roll).strip()
            else None
        )
        name = read.get("name")
        sub.student_name = str(name).strip()[:120] if name else None

        reasons, review_items = [], []
        unreadable_found = False
        rag_index, rag_error = None, None
        if any(q.type == "Descriptive" for q in questions):
            try:
                rag_index = KnowledgeIndex.load(db, exam.id, get_embedder())
            except Exception as exc:  # no embedder / model mismatch: descriptive answers go to the teacher
                rag_error = str(exc)
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

        if second_error is not None:
            reasons.append(f"The second reading failed: {second_error[:150]}")
            review_items.append("Second reading")
        if second is not None and sub.roll_no and _roll_key(second.get("roll_no")) != _roll_key(sub.roll_no):
            reasons.append("Roll number differs between two readings")
            review_items.append("Roll number")

        # Keep the first valid occurrence. Conflicting duplicate question numbers
        # are treated as unreadable instead of silently accepting the last value.
        expected = {q.number for q in questions}
        found, duplicate_numbers = _collect(raw_answers, expected)
        if second is not None:
            found2, _ = _collect(second["answers"], expected)
            disagree = {q.number for q in questions if not _same_reading(found.get(q.number), found2.get(q.number), q.type)}
            if disagree:
                reasons.append("Two readings of some answers disagree")
                duplicate_numbers = duplicate_numbers | disagree  # treated exactly like an unreadable answer

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

            extra = {}
            if (q.type == "Descriptive" and given is not None and str(given).strip()
                    and confidence >= settings.MIN_CONFIDENCE):
                graded = _grade_descriptive(q, str(given), rag_index, rag_error)
                result, marks = graded["result"], graded["marks"]
                extra = {
                    "feedback": graded["feedback"] or None,
                    "citations_json": json.dumps(graded["citations"]) if graded["citations"] else None,
                    "retrieval_score": graded["retrieval_score"],
                    "needs_review": graded["needs_review"],
                }
                if graded["needs_review"]:
                    review_items.append(f"Q{q.number}")
                    reasons.append(f"Q{q.number} AI grade needs a teacher check: {graded['review_reason']}")
            else:
                result, marks = scoring.score_answer(
                    q, given, confidence, exam.negative_marking
                )
                if result == "unreadable":
                    review_items.append(f"Q{q.number}")
                    unreadable_found = True
            total += marks
            answer_rows.append(
                Answer(
                    submission_id=sub.id,
                    question_number=q.number,
                    student_answer="" if given is None else str(given).strip()[:4000],
                    correct_answer=q.correct_answer,
                    result=result,
                    marks=marks,
                    confidence=confidence,
                    **extra,
                )
            )

        db.add_all(answer_rows)
        sub.total_score = max(0.0, total)

        if review_items:
            if unreadable_found:
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
