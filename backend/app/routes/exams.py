import csv
import io
import json
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Response, UploadFile
from openpyxl import load_workbook
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from app.core.access import optional_user, owned_exam, owned_submission
from app.core.config import settings
from app.core.database import get_db
from app.models import Exam, PaperCache, Question, Submission, User
from app.services import llm_service, question_paper_reader, question_parser
from app.services.answer_key_reader import AnswerKeyReadError, extract_answer_key
from app.services.export import build_workbook
from app.services.processing import MIME, recover_stuck, run_batch

router = APIRouter(prefix="/exams", tags=["exams"])
FILE_TYPES = {".pdf", ".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------- helpers
# The first bytes of each real file type. An extension alone proves nothing, so the content is checked too.
SIGNATURES = {".jpg": b"\xff\xd8\xff", ".jpeg": b"\xff\xd8\xff", ".png": b"\x89PNG\r\n\x1a\n", ".xlsx": b"PK\x03\x04", ".docx": b"PK\x03\x04"}


def looks_like(suffix: str, data: bytes) -> bool:
    if suffix == ".pdf":
        return b"%PDF" in data[:1024]  # the PDF spec allows a little junk before the header
    if suffix == ".csv":
        return b"\x00" not in data[:2048]  # text, not a binary file renamed to .csv
    signature = SIGNATURES.get(suffix)
    return signature is None or data.startswith(signature)


def read_upload(f: UploadFile, allowed: set[str]) -> tuple[bytes, str]:
    suffix = Path(f.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"{f.filename}: unsupported file type. Allowed: {', '.join(sorted(allowed))}")
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    data = f.file.read(limit + 1)  # never pulls more than the limit into memory
    if len(data) > limit:
        raise HTTPException(400, f"{f.filename}: larger than {settings.MAX_UPLOAD_MB} MB")
    if not data:
        raise HTTPException(400, f"{f.filename}: the file is empty")
    if not looks_like(suffix, data):
        raise HTTPException(400, f"{f.filename}: the content is not a real {suffix} file")
    return data, suffix


def get_exam(db: Session, exam_id: int, user: User | None = None) -> Exam:
    """The exam, only if it belongs to the signed-in teacher (see app/core/access.py)."""
    return owned_exam(db, exam_id, user)


def key_from_table(rows) -> dict:
    """Read question-number -> answer pairs from the first two columns of a CSV/Excel table."""
    out = {}
    for row in rows:
        if len(row) < 2 or row[0] is None:
            continue
        m = re.match(r"\s*Q?\s*(\d+)", str(row[0]), re.I)
        if m and row[1] is not None and str(row[1]).strip() != "":
            val = row[1]
            out[m.group(1)] = str(int(val)) if isinstance(val, float) and val == int(val) else str(val).strip()
    return out


def results_for(exam: Exam) -> list[dict]:
    rows = []
    for s in exam.submissions:
        if s.status != "done":
            continue
        ans = [{"number": a.question_number, "student_answer": a.student_answer, "correct_answer": a.correct_answer,
                "result": a.result, "marks": a.marks, "feedback": a.feedback,
                "citations": json.loads(a.citations_json) if a.citations_json else [],
                "needs_review": bool(a.needs_review)} for a in s.answers]
        rows.append({
            "id": s.id, "roll_no": s.roll_no or "-", "name": s.student_name or "-", "total": s.total_score or 0,
            "pct": round((s.total_score or 0) / exam.total_marks * 100) if exam.total_marks else 0,
            # Total-only manual marks have no question-level grading data.
            "correct": sum(a["result"] == "correct" for a in ans) if ans else None,
            "wrong": sum(a["result"] == "wrong" for a in ans) if ans else None,
            "unanswered": sum(a["result"] == "unanswered" for a in ans) if ans else None,
            "manual": s.manual, "answers": ans,
        })
    return sorted(rows, key=lambda r: str(r["roll_no"]))


def review_for(exam: Exam) -> list[dict]:
    return [{"id": s.id, "roll_no": s.roll_no, "filename": s.filename, "name": s.student_name, "detail": s.review_detail or "",
             "reason": s.review_reason or ""} for s in exam.submissions if s.status == "needs_review"]


# ---------------------------------------------------------------- step 1-3: read paper and key (nothing saved yet)
@router.post("/parse-paper")
def parse_paper(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Cheapest route first: cache -> local text/OCR + rules -> the LLM on the text -> the LLM on the file."""
    data, suffix = read_upload(file, FILE_TYPES)
    doc = question_paper_reader.read_paper(data, suffix)
    pages = doc["pages"]

    cached = db.get(PaperCache, doc["file_hash"])
    cached_result = None
    if cached:
        try:
            cached_result = json.loads(cached.result_json)
        except ValueError:
            cached_result = None
        # Papers cached by an older classifier carry wrong question types: ignore them and read the paper again.
        if cached_result is not None and cached_result.get("parser_version") != question_parser.PARSER_VERSION:
            cached_result = None
    if cached_result is not None:
        return {**cached_result, "source": "cache", "pages": pages}

    result, source = None, ""
    if doc["usable"]:
        result, source = question_parser.parse_rules(doc["text"]), "rules"
        if result is None and settings.LLM_PROVIDER != "mock":
            try:
                result, source = llm_service.parse_paper_text(doc["text"]), "llm-text"
            except llm_service.LLMError:
                result = None
    if result is None:  # no usable local text (no OCR installed, diagram-only pages, ...): let the vision model look at the file
        try:
            result = llm_service.parse_paper(data, MIME[suffix])
            source = "mock" if settings.LLM_PROVIDER == "mock" else "llm-file"
        except llm_service.LLMError as exc:
            raise HTTPException(502, str(exc))

    result = question_parser.normalise(result)
    if result["questions"] and source != "mock":
        db.merge(PaperCache(file_hash=doc["file_hash"], result_json=json.dumps(result), source=source))
        db.commit()
    return {**result, "source": source, "pages": pages}


@router.post("/parse-key")
def parse_key(file: UploadFile = File(...)):
    data, suffix = read_upload(file, FILE_TYPES | {".csv", ".xlsx"})
    if suffix == ".csv":
        text = data.decode("utf-8-sig", errors="replace")
        answers = key_from_table(csv.reader(io.StringIO(text)))
    elif suffix == ".xlsx":
        ws = load_workbook(io.BytesIO(data), data_only=True).active
        answers = key_from_table(ws.iter_rows(values_only=True))
    else:
        try:
            answers, source, ocr_note = extract_answer_key(data, suffix)
        except AnswerKeyReadError as exc:
            raise HTTPException(
                422,
                f"Could not extract answers from this file. {exc} "
                "Try a clearer image, a numbered key, or lines such as 'Answer = 3601'.",
            ) from exc
        if not answers:
            raise HTTPException(
                422,
                "No answers detected. Use numbered rows such as '1. 3601' or solved-work lines such as 'Answer = 3601'.",
            )
        return {
            "answers": {str(k): str(v) for k, v in answers.items()},
            "source": source,
            "ocr_note": ocr_note,
        }
    if not answers:
        raise HTTPException(422, "No answers found. Use two columns: question number, correct answer.")
    return {"answers": {str(k): str(v) for k, v in answers.items()}, "source": "table"}


# ---------------------------------------------------------------- create exam and upload sheets
class RubricItem(BaseModel):
    criterion: str
    marks: float


SUPPORTED_QUESTION_TYPES = {"MCQ", "Numeric", "FillBlank", "ShortAnswer", "TrueFalse", "Formula", "Date", "Matching", "Descriptive"}

class QuestionIn(BaseModel):
    number: int
    text: str = ""
    type: str = "MCQ"
    marks: float
    correct_answer: str = ""
    rubric: list[RubricItem] | None = None  # Descriptive questions
    model_answer: str = ""  # Descriptive questions

    @field_validator("type", mode="before")
    @classmethod
    def validate_question_type(cls, value):
        aliases = {"fill in the blanks": "FillBlank", "fill-in-the-blanks": "FillBlank", "short answer": "ShortAnswer", "true/false": "TrueFalse"}
        raw = str(value or "MCQ").strip()
        canonical = aliases.get(raw.casefold(), raw)
        if canonical not in SUPPORTED_QUESTION_TYPES:
            raise ValueError(f"Unsupported question type: {raw}")
        return canonical


class ExamIn(BaseModel):
    title: str
    negative_marking: float = 0
    questions: list[QuestionIn]


@router.post("")
def create_exam(body: ExamIn, db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    if not body.questions:
        raise HTTPException(422, "The exam has no questions")
    missing = [f"Q{q.number}" for q in body.questions if q.type != "Descriptive" and not q.correct_answer.strip()]
    if missing:
        raise HTTPException(422, f"Answer key is missing for: {', '.join(missing)}")
    for q in body.questions:
        if q.type != "Descriptive":
            continue
        # Descriptive questions may rely on uploaded reference books instead of a model answer.
        # If no reference evidence is available during grading, the grader safely sends the answer to review.
        if q.rubric:
            if any(r.marks <= 0 or not r.criterion.strip() for r in q.rubric):
                raise HTTPException(422, f"Q{q.number}: every rubric line needs a name and marks above zero.")
            if abs(sum(r.marks for r in q.rubric) - q.marks) > 1e-6:
                raise HTTPException(422, f"Q{q.number}: rubric marks add up to {sum(r.marks for r in q.rubric):g}, not {q.marks:g}.")
    exam = Exam(title=body.title.strip() or "Untitled exam", negative_marking=min(0, body.negative_marking),
                total_marks=sum(q.marks for q in body.questions), owner_id=user.id if user else None)
    exam.questions = [Question(number=q.number, text=q.text, type=q.type, marks=q.marks, correct_answer=q.correct_answer.strip(),
                               rubric_json=json.dumps([r.model_dump() for r in q.rubric]) if q.rubric else None,
                               model_answer=(q.model_answer.strip() or q.correct_answer.strip()) if q.type == "Descriptive" else q.model_answer.strip())
                      for q in body.questions]
    db.add(exam)
    db.commit()
    return {"id": exam.id}


@router.post("/{exam_id}/sheets")
def upload_sheets(exam_id: int, background: BackgroundTasks, files: list[UploadFile] = File(...), db: Session = Depends(get_db),
                  user: User | None = Depends(optional_user)):
    exam = get_exam(db, exam_id, user)
    folder = settings.UPLOAD_DIR / str(exam.id)
    folder.mkdir(parents=True, exist_ok=True)
    subs = []
    for f in files:
        data, suffix = read_upload(f, FILE_TYPES)
        path = folder / f"{uuid.uuid4().hex}{suffix}"  # generated name: no overwrites, no path tricks
        path.write_bytes(data)
        subs.append(Submission(exam_id=exam.id, filename=Path(f.filename).name, stored_path=str(path)))
    db.add_all(subs)
    db.commit()
    background.add_task(run_batch, [s.id for s in subs])
    return {"queued": len(subs)}


# ---------------------------------------------------------------- read results
@router.get("")
def list_exams(db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    query = db.query(Exam)
    if user is not None:
        query = query.filter(Exam.owner_id == user.id)
    exams = query.order_by(Exam.id.desc()).all()
    return [{"id": e.id, "title": e.title, "sheets": len(e.submissions)} for e in exams]


@router.get("/{exam_id}/status")
def status(exam_id: int, db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    exam = get_exam(db, exam_id, user)
    return [{"id": s.id, "filename": s.filename, "roll_no": s.roll_no, "status": s.status} for s in exam.submissions]


@router.get("/{exam_id}/results")
def results(exam_id: int, db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    exam = get_exam(db, exam_id, user)
    rows = results_for(exam)
    pcts = [r["pct"] for r in rows]
    return {
        "exam": {"id": exam.id, "title": exam.title, "total_marks": exam.total_marks, "negative_marking": exam.negative_marking},
        "stats": {"students": len(rows), "average": round(sum(pcts) / len(pcts)) if pcts else 0,
                  "highest": max(pcts, default=0), "lowest": min(pcts, default=0), "needs_review": len(review_for(exam))},
        "rows": rows,
    }


@router.get("/{exam_id}/unreadable")
def unreadable(exam_id: int, db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    return review_for(get_exam(db, exam_id, user))


@router.get("/{exam_id}/export")
def export(exam_id: int, db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    exam = get_exam(db, exam_id, user)
    blob = build_workbook(exam, results_for(exam), review_for(exam))
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", exam.title) or "results"
    return Response(blob, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{name}_results.xlsx"'})


# ---------------------------------------------------------------- teacher enters marks for an unreadable sheet
class ManualIn(BaseModel):
    marks: float
    name: str | None = None
    roll_no: str | None = None


@router.post("/submissions/{sub_id}/manual")
def manual_marks(sub_id: int, body: ManualIn, db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    """Save a teacher-entered total and optional identity details.

    A total-only manual mark deliberately does not fabricate question-level
    correct/wrong counts. Those are shown as unavailable in the results table.
    """
    sub = owned_submission(db, sub_id, user)
    total = sub.exam.total_marks
    if not 0 <= body.marks <= total:
        raise HTTPException(422, f"Marks must be between 0 and {total}")
    if body.name is not None:
        sub.student_name = body.name.strip()[:120] or None
    if body.roll_no is not None:
        sub.roll_no = body.roll_no.strip()[:50] or None
    sub.total_score = float(body.marks)
    sub.manual = True
    sub.status = "done"
    sub.review_reason = None
    sub.review_detail = None
    db.commit()
    return {"ok": True, "manual_total_only": len(sub.answers) == 0}


# ---------------------------------------------------------------- sheets that never finished
@router.post("/{exam_id}/retry-stuck")
def retry_stuck(exam_id: int, background: BackgroundTasks, db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    """Put sheets stuck in 'queued' or 'reading' (for example after the server stopped mid-batch) back to work."""
    exam = get_exam(db, exam_id, user)
    ids = recover_stuck(db, exam.id)
    if ids:
        background.add_task(run_batch, ids)
    return {"requeued": len(ids)}

