import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.auth_guard import AuthGuardMiddleware
from app.core.database import SessionLocal, init_db
from app.routes import auth, exams, knowledge, system
from app.services import processing


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_db()  # creates missing PostgreSQL tables on first run
    if settings.RECOVER_ON_START:
        # Sheets left 'queued' or 'reading' by a stopped server would otherwise wait forever.
        db = SessionLocal()
        try:
            ids = processing.recover_stuck(db)
        finally:
            db.close()
        if ids:
            threading.Thread(target=processing.run_batch, args=(ids,), daemon=True, name="recover-stuck").start()
    yield


app = FastAPI(title="Handwriting Answer Evaluation API", description="LLM-based answer sheet checking and scoring", version="2.0", lifespan=lifespan)
app.add_middleware(AuthGuardMiddleware)  # login required for /api (added before CORS so CORS headers stay on 401 replies)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # lets frontend/index.html work when opened as a file

app.include_router(auth.router, prefix="/api")
app.include_router(system.router, prefix="/api")
app.include_router(exams.router, prefix="/api")
app.include_router(knowledge.router, prefix="/api")

# The UI is served from the same server: open http://127.0.0.1:8000
if settings.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=settings.FRONTEND_DIR, html=True), name="frontend")
