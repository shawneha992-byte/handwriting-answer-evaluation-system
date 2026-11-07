"""Gate for the API: every /api route needs a valid login token, except sign-in, sign-up and health.

It is a middleware, so none of the existing route files had to change.
Set AUTH_REQUIRED=false in .env to switch the gate off (for example in automated tests).
"""
from starlette.responses import JSONResponse

from app.core.config import settings
from app.core.security import decode_token

PUBLIC_PATHS = {"/api/health", "/api/auth/login", "/api/auth/register"}


def bearer_token(headers) -> str | None:
    value = headers.get("authorization", "")
    scheme, _, token = value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


class AuthGuardMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and settings.AUTH_REQUIRED:
            path = scope["path"].rstrip("/") or "/"
            if path.startswith("/api") and scope["method"] != "OPTIONS" and path not in PUBLIC_PATHS:
                headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
                if decode_token(bearer_token(headers)) is None:
                    response = JSONResponse({"detail": "Please sign in to continue."}, status_code=401, headers={"WWW-Authenticate": "Bearer"})
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)
