"""Split extracted text into overlapping, sentence-aware chunks that remember their page."""
import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
    """Greedy sentence packing. The last sentences of a chunk are repeated at the start of the next one."""
    max_chars = max(100, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars // 2))
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in _sentences(text):
        # A single very long sentence is cut hard so no chunk exceeds max_chars.
        while len(sentence) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current, size = [], 0
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars - overlap:] if overlap else sentence[max_chars:]
        if size + len(sentence) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            tail: list[str] = []
            tail_size = 0
            for prev in reversed(current):
                if tail_size + len(prev) + 1 > overlap:
                    break
                tail.insert(0, prev)
                tail_size += len(prev) + 1
            current, size = tail, tail_size
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def chunk_pages(pages: list[tuple[int, str]], max_chars: int = 900, overlap: int = 150) -> list[dict]:
    """pages = [(page_number, text)]. Returns [{"page", "idx", "text"}] with idx running across the document."""
    out: list[dict] = []
    for page, text in pages:
        for piece in chunk_text(text, max_chars, overlap):
            out.append({"page": page, "idx": len(out), "text": piece})
    return out
