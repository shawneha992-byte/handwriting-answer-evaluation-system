"""Who is calling, and which exams they may touch.

Every exam belongs to the teacher who created it. A teacher asking for someone else's exam gets the same
"not found" answer as for an exam that does not exist, so exam IDs cannot be probed.
When AUTH_REQUIRED is off (automated tests, a local demo) there is no signed-in user and no ownership check.
"""
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.auth_guard import bearer_token
from app.core.config import settings
from app.core.database import get_db
from app.core.security import decode_token
from app.models import Exam, Submission, User


def optional_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    if not settings.AUTH_REQUIRED:
        return None
    data = decode_token(bearer_token(request.headers))
    user = db.get(User, int(data["sub"])) if data else None
    if not user or not user.is_active:
        raise HTTPException(401, "Please sign in to continue.")
    return user


def owned_exam(db: Session, exam_id: int, user: User | None) -> Exam:
    exam = db.get(Exam, exam_id)
    if exam is None or (user is not None and exam.owner_id != user.id):
        raise HTTPException(404, "Exam not found")
    return exam


def owned_submission(db: Session, sub_id: int, user: User | None) -> Submission:
    sub = db.get(Submission, sub_id)
    if sub is None or (user is not None and sub.exam.owner_id != user.id):
        raise HTTPException(404, "Sheet not found")
    return sub
