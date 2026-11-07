"""Login / registration layer: sign up, sign in, protected routes, token rules."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{tmp}/auth_test.db")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("UPLOAD_DIR", f"{tmp}/uploads")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core import security  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.routes import auth as auth_routes  # noqa: E402

GOOD = {"full_name": "Asha Verma", "email": "asha@example.com", "password": "S3cure-pass!", "institution": "Greenfield School"}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    auth_routes._fails.clear()
    auth_routes._asks.clear()
    with TestClient(app) as c:
        yield c


def test_api_is_locked_until_login(client):
    assert client.get("/api/health").status_code == 200  # health stays public
    for path in ("/api/exams", "/api/dashboard", "/api/config", "/api/auth/me"):
        r = client.get(path)
        assert r.status_code == 401, path
    assert client.get("/api/exams", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    assert client.get("/").status_code in (200, 404)  # the page itself (login screen) is never blocked


def test_register_login_and_use_api(client):
    r = client.post("/api/auth/register", json=GOOD)
    assert r.status_code == 201
    body = r.json()
    assert body["user"]["email"] == "asha@example.com" and "password" not in str(body)
    h = {"Authorization": f"Bearer {body['token']}"}

    assert client.get("/api/auth/me", headers=h).json()["full_name"] == "Asha Verma"
    assert client.get("/api/exams", headers=h).status_code == 200
    assert client.get("/api/dashboard", headers=h).status_code == 200

    # duplicate email (any capitalisation) is refused
    dup = client.post("/api/auth/register", json={**GOOD, "email": "ASHA@Example.com"})
    assert dup.status_code == 409

    ok = client.post("/api/auth/login", json={"email": " Asha@Example.com ", "password": GOOD["password"]})
    assert ok.status_code == 200 and ok.json()["token"]
    assert client.post("/api/auth/login", json={"email": "asha@example.com", "password": "wrong-pass"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "whatever1"}).status_code == 401


@pytest.mark.parametrize("patch,code", [
    ({"full_name": "A"}, 422), ({"email": "not-an-email"}, 422), ({"password": "short"}, 422), ({"password": "x" * 200}, 422),
])
def test_register_validation(client, patch, code):
    r = client.post("/api/auth/register", json={**GOOD, "email": "v@example.com", **patch})
    assert r.status_code == code and isinstance(r.json()["detail"], str)


def test_remember_me_makes_longer_token(client):
    client.post("/api/auth/register", json={**GOOD, "email": "long@example.com"})
    short = client.post("/api/auth/login", json={"email": "long@example.com", "password": GOOD["password"]}).json()
    long_ = client.post("/api/auth/login", json={"email": "long@example.com", "password": GOOD["password"], "remember": True}).json()
    assert long_["expires_in"] > short["expires_in"]


def test_tampered_and_expired_tokens_are_rejected(client):
    token = client.post("/api/auth/register", json={**GOOD, "email": "t@example.com"}).json()["token"]
    head, payload, sig = token.split(".")
    forged = f"{head}.{security._b64(b'{\"sub\":\"1\",\"exp\":9999999999}')}.{sig}"
    assert security.decode_token(forged) is None
    assert security.decode_token(security.create_token(1, "t@example.com", -10)) is None
    assert client.get("/api/exams", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_repeated_failures_are_throttled(client):
    client.post("/api/auth/register", json={**GOOD, "email": "brute@example.com"})
    codes = [client.post("/api/auth/login", json={"email": "brute@example.com", "password": f"bad-pass-{i}"}).status_code for i in range(10)]
    assert codes[:8] == [401] * 8 and codes[8] == 429
    # even the right password waits until the window passes
    assert client.post("/api/auth/login", json={"email": "brute@example.com", "password": GOOD["password"]}).status_code == 429


def test_password_hash_is_salted_and_verifiable():
    a, b = security.hash_password("hello-world"), security.hash_password("hello-world")
    assert a != b and security.verify_password("hello-world", a) and not security.verify_password("Hello-world", a)
    assert not security.verify_password("x", "garbage")


# ---------------------------------------------------------------- forgot / reset password
@pytest.fixture()
def mailbox(monkeypatch):
    sent = []
    monkeypatch.setattr(auth_routes, "send_reset_email", lambda to, name, link, minutes: sent.append((to, link)) or True)
    return sent


def _token_from(link):
    return link.split("token=", 1)[1]


def test_forgot_password_does_not_reveal_whether_an_email_exists(client, mailbox):
    client.post("/api/auth/register", json={**GOOD, "email": "known@example.com"})
    a = client.post("/api/auth/forgot-password", json={"email": "known@example.com"})
    b = client.post("/api/auth/forgot-password", json={"email": "ghost@example.com"})
    assert a.status_code == b.status_code == 200 and a.json() == b.json()
    assert [to for to, _ in mailbox] == ["known@example.com"]  # only the real account got mail
    assert client.post("/api/auth/forgot-password", json={"email": "bad"}).status_code == 422


def test_reset_password_flow_and_single_use(client, mailbox):
    client.post("/api/auth/register", json={**GOOD, "email": "reset@example.com"})
    client.post("/api/auth/forgot-password", json={"email": "reset@example.com"})
    token = _token_from(mailbox[-1][1])

    assert client.post("/api/auth/reset-password", json={"token": token, "password": "short"}).status_code == 422
    ok = client.post("/api/auth/reset-password", json={"token": token, "password": "Brand-new-pass1"})
    assert ok.status_code == 200
    assert client.post("/api/auth/login", json={"email": "reset@example.com", "password": GOOD["password"]}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "reset@example.com", "password": "Brand-new-pass1"}).status_code == 200
    # the same link cannot be used twice
    again = client.post("/api/auth/reset-password", json={"token": token, "password": "Another-pass-2"})
    assert again.status_code == 400


def test_reset_and_login_tokens_cannot_be_swapped(client, mailbox):
    login_token = client.post("/api/auth/register", json={**GOOD, "email": "swap@example.com"}).json()["token"]
    client.post("/api/auth/forgot-password", json={"email": "swap@example.com"})
    reset_token = _token_from(mailbox[-1][1])
    # a login token cannot reset a password, and a reset token cannot open the API
    assert client.post("/api/auth/reset-password", json={"token": login_token, "password": "Whatever-pass1"}).status_code == 400
    assert client.get("/api/exams", headers={"Authorization": f"Bearer {reset_token}"}).status_code == 401
    assert client.post("/api/auth/reset-password", json={"token": "garbage", "password": "Whatever-pass1"}).status_code == 400


def test_expired_reset_link_is_rejected(client):
    client.post("/api/auth/register", json={**GOOD, "email": "old@example.com"})
    from app.core.database import SessionLocal
    from app.models import User
    with SessionLocal() as db:
        u = db.query(User).filter(User.email == "old@example.com").one()
        stale = security.create_token(u.id, u.email, -5, purpose="reset", extra={"pwd": security.password_fingerprint(u.password_hash)})
    assert client.post("/api/auth/reset-password", json={"token": stale, "password": "Valid-pass-123"}).status_code == 400


def test_reset_requests_are_throttled(client, mailbox):
    client.post("/api/auth/register", json={**GOOD, "email": "spam@example.com"})
    codes = [client.post("/api/auth/forgot-password", json={"email": "spam@example.com"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429] and len(mailbox) == 3


def test_mailer_prints_link_when_smtp_is_not_configured(monkeypatch, caplog):
    from app.core import mailer
    monkeypatch.setattr(settings, "SMTP_HOST", "")
    with caplog.at_level("WARNING", logger="uvicorn.error"):
        assert mailer.send_reset_email("a@example.com", "Asha Verma", "http://x/#reset?token=abc", 30) is False
    assert "http://x/#reset?token=abc" in caplog.text
