"""Password hashing and signed login tokens.

Uses only the Python standard library (scrypt + HMAC-SHA256), so no new packages are needed.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from functools import lru_cache

from app.core.config import BASE_DIR, settings

_N, _R, _P = 2**14, 8, 1  # scrypt cost: about 16 MB of memory per hash


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


# ---------------------------------------------------------------- passwords
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = _unb64(digest)
        actual = hashlib.scrypt(password.encode(), salt=_unb64(salt), n=int(n), r=int(r), p=int(p), dklen=len(expected))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------- signed tokens (HS256 JWT)
@lru_cache(maxsize=1)
def _secret() -> bytes:
    """AUTH_SECRET_KEY from .env, or a random key created once and kept in backend/.auth_secret."""
    if settings.AUTH_SECRET_KEY:
        return settings.AUTH_SECRET_KEY.encode()
    path = BASE_DIR / ".auth_secret"
    try:
        if path.exists() and path.read_text().strip():
            return path.read_text().strip().encode()
        key = secrets.token_urlsafe(48)
        path.write_text(key)
        return key.encode()
    except OSError:  # read-only disk: fall back to a per-process key (logins reset on restart)
        return secrets.token_urlsafe(48).encode()


def _sign(message: str) -> str:
    return _b64(hmac.new(_secret(), message.encode(), hashlib.sha256).digest())


def create_token(user_id: int, email: str, lifetime_seconds: int) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    now = int(time.time())
    payload = _b64(json.dumps({"sub": str(user_id), "email": email, "iat": now, "exp": now + lifetime_seconds}, separators=(",", ":")).encode())
    return f"{header}.{payload}.{_sign(f'{header}.{payload}')}"


def decode_token(token: str | None) -> dict | None:
    """Return the token payload, or None if it is missing, tampered with or expired."""
    if not token:
        return None
    try:
        header, payload, signature = token.split(".")
        if not hmac.compare_digest(signature, _sign(f"{header}.{payload}")):
            return None
        data = json.loads(_unb64(payload))
        if int(data["exp"]) < time.time():
            return None
        return data
    except (ValueError, KeyError, TypeError):
        return None
