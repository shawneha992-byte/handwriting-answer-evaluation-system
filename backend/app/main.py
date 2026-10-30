from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import init_db
from app.routes import exams, system


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_db()  # creates the MySQL database and tables on first run
    yield


app = FastAPI(title="Handwriting Answer Evaluation API", description="LLM-based answer sheet checking and scoring", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # lets frontend/index.html work when opened as a file

app.include_router(system.router, prefix="/api")
app.include_router(exams.router, prefix="/api")

# The UI is served from the same server: open http://127.0.0.1:8000
if settings.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=settings.FRONTEND_DIR, html=True), name="frontend")
