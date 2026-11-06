"""Reads text out of a question paper WITHOUT calling any paid/limited AI.

Digital PDF page  -> text is copied directly with PyMuPDF (instant, exact).
Scanned page/image -> text is read with Tesseract OCR (free, runs on your own computer).

If Tesseract is not installed the app still works: the caller falls back to the configured local vision model.
"""
import hashlib
import re
from functools import lru_cache
from io import BytesIO

from app.core.config import settings

try:
    import pymupdf as fitz  # PyMuPDF
except ImportError:
    try:
        import fitz  # older PyMuPDF versions use this name
    except ImportError:  # keep the app importable even if the package is missing
        fitz = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n").replace("\x0c", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


@lru_cache(maxsize=1)
def ocr_available() -> bool:
    """True only if pytesseract AND the Tesseract program are both installed."""
    if pytesseract is None:
        return False
    if settings.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr(image) -> str:
    return clean_text(pytesseract.image_to_string(image, config="--psm 3"))


def read_paper(data: bytes, suffix: str) -> dict:
    """Returns {"text", "pages": [{page, method, characters, needs_review}], "file_hash", "usable"}.

    usable=False means we could not get trustworthy text locally (no OCR installed, corrupt file, ...),
    so the caller should let the configured vision model inspect the original file instead.
    """
    out = {"text": "", "pages": [], "file_hash": file_hash(data), "usable": False}
    texts: list[str] = []
    try:
        if suffix == ".pdf":
            if fitz is None:
                return out
            doc = fitz.open(stream=data, filetype="pdf")
            try:
                for n, page in enumerate(doc, start=1):
                    text, method = clean_text(page.get_text("text")), "digital_pdf"
                    if len(text) < settings.MIN_PAGE_TEXT_CHARS and ocr_available():
                        pix = page.get_pixmap(matrix=fitz.Matrix(settings.OCR_ZOOM, settings.OCR_ZOOM), alpha=False)
                        text, method = _ocr(Image.open(BytesIO(pix.tobytes("png")))), "ocr"
                    texts.append(text)
                    out["pages"].append({"page": n, "method": method, "characters": len(text),
                                         "needs_review": len(text) < settings.MIN_PAGE_TEXT_CHARS})
            finally:
                doc.close()
        elif ocr_available():  # a photo/scan of the paper
            text = _ocr(Image.open(BytesIO(data)))
            texts.append(text)
            out["pages"].append({"page": 1, "method": "ocr", "characters": len(text),
                                 "needs_review": len(text) < settings.MIN_PAGE_TEXT_CHARS})
    except Exception:  # corrupt PDF, unreadable image, OCR crash: let the caller fall back
        out["pages"] = []
        return out
    out["text"] = "\n".join(texts)
    # usable only when every page produced real text (a page with only a diagram is flagged, not hidden)
    out["usable"] = bool(out["pages"]) and not any(p["needs_review"] for p in out["pages"])
    return out
