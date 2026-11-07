"""Sign-up, sign-in and 'who am I'. Only the authentication layer: exam logic is untouched."""
import re
import time
from collections import defaultdict
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.auth_guard import bearer_token
from app.core.config import settings
from app.core.database import get_db
from app.core.security import create_token, decode_token, hash_password, verify_password
from app.models import User

router = APIRouter(prefix="/auth", tags=["auth"])

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")
REMEMBER_DAYS = 30

# Simple brute-force brake: too many wrong passwords from one address/email pair pauses sign-in for a while.
_MAX_FAILS, _WINDOW_SECONDS = 8, 15 * 60
_fails: dict[tuple, list[float]] = defaultdict(list)
_DUMMY_HASH = hash_password("not-a-real-password")  # makes unknown emails take as long as wrong passwords


class RegisterIn(BaseModel):
    full_name: str
    email: str
    password: str
    institution: str | None = None


class LoginIn(BaseModel):
    email: str
    password: str
    remember: bool = False


def public_user(user: User) -> dict:
    return {"id": user.id, "full_name": user.full_name, "email": user.email, "institution": user.institution}


def token_response(user: User, seconds: int) -> dict:
    return {"token": create_token(user.id, user.email, seconds), "token_type": "bearer", "expires_in": seconds, "user": public_user(user)}


def _throttle_key(request: Request, email: str) -> tuple:
    return (request.client.host if request.client else "?", email)


def _recent_fails(key: tuple) -> list[float]:
    now = time.time()
    _fails[key] = [t for t in _fails[key] if now - t < _WINDOW_SECONDS]
    return _fails[key]


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Dependency: the signed-in user. Use it on any future route that needs to know who is calling."""
    data = decode_token(bearer_token(request.headers))
    user = db.get(User, int(data["sub"])) if data else None
    if not user or not user.is_active:
        raise HTTPException(401, "Please sign in to continue.")
    return user


@router.post("/register", status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    name = " ".join(body.full_name.split())
    email = body.email.strip().lower()
    if len(name) < 2 or len(name) > 120:
        raise HTTPException(422, "Please enter your full name.")
    if not EMAIL_RE.match(email) or len(email) > 254:
        raise HTTPException(422, "Please enter a valid email address.")
    if len(body.password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters.")
    if len(body.password) > 128:
        raise HTTPException(422, "Password must be 128 characters or fewer.")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(409, "An account with this email already exists. Try signing in.")

    user = User(full_name=name, email=email, institution=(body.institution or "").strip()[:150] or None, password_hash=hash_password(body.password), last_login_at=datetime.utcnow())
    db.add(user)
    try:
        db.commit()
    except IntegrityError:  # two sign-ups with the same email at the same moment
        db.rollback()
        raise HTTPException(409, "An account with this email already exists. Try signing in.")
    db.refresh(user)
    return token_response(user, settings.AUTH_TOKEN_HOURS * 3600)


@router.post("/login")
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    email = body.email.strip().lower()
    key = _throttle_key(request, email)
    if len(_recent_fails(key)) >= _MAX_FAILS:
        raise HTTPException(429, "Too many failed attempts. Please wait a few minutes and try again.")

    user = db.query(User).filter(User.email == email).first()
    ok = verify_password(body.password, user.password_hash if user else _DUMMY_HASH)
    if not user or not ok or not user.is_active:
        _fails[key].append(time.time())
        raise HTTPException(401, "Incorrect email or password.")

    _fails.pop(key, None)
    user.last_login_at = datetime.utcnow()
    db.commit()
    seconds = REMEMBER_DAYS * 86400 if body.remember else settings.AUTH_TOKEN_HOURS * 3600
    return token_response(user, seconds)


@router.get("/me")
def me(user: User = Depends(current_user)):
    return public_user(user)
