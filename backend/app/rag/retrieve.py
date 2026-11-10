"""Question-scoped hybrid retrieval using PostgreSQL pgvector + keyword ranking.

PostgreSQL performs vector nearest-neighbour search in the database. SQLite remains supported
only for unit tests, using the legacy serialized vectors and NumPy cosine similarity.
"""
import math
import re
from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from app.rag.models import KnowledgeChunk, KnowledgeDoc

RRF_K = 60
_STOP = set("a an the of and or to in on for with is are was were be by as at it this that from which what how why "
            "explain describe define write give state discuss".split())


class RagError(Exception):
    """Retrieval could not run (for example vectors came from another embedding model)."""


@dataclass
class Hit:
    chunk_id: int
    source: str
    page: int
    text: str
    vector_score: float
    fused: float


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


class KnowledgeIndex:
    def __init__(self, db: Session, exam_id: int, rows: list[KnowledgeChunk], names: dict[int, str], embedder):
        self.db, self.exam_id = db, exam_id
        self.embedder, self.rows, self.names = embedder, rows, names
        self.by_id = {r.id: r for r in rows}
        self.docs_tokens = [tokens(r.text) for r in rows]
        self.avg_len = sum(len(t) for t in self.docs_tokens) / max(1, len(self.docs_tokens))
        df: dict[str, int] = {}
        for toks in self.docs_tokens:
            for tok in set(toks):
                df[tok] = df.get(tok, 0) + 1
        n = len(rows)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.is_postgres = self.db.get_bind().dialect.name == "postgresql"
        self.matrix = None
        if not self.is_postgres:
            self.matrix = np.vstack([np.frombuffer(r.embedding, dtype=np.float32) for r in rows])

    @classmethod
    def load(cls, db: Session, exam_id: int, embedder) -> "KnowledgeIndex | None":
        rows = db.query(KnowledgeChunk).filter(KnowledgeChunk.exam_id == exam_id).all()
        if not rows:
            return None
        matching = [r for r in rows if r.embed_model == embedder.name]
        if not matching:
            raise RagError("Reference material was embedded with a different model. Re-upload/re-index it with the configured embedding model.")
        if db.get_bind().dialect.name == "postgresql":
            missing = [r for r in matching if r.embedding_vector is None]
            if missing:
                raise RagError(
                    f"{len(missing)} stored chunks do not yet have pgvector embeddings. "
                    "Use the Re-index knowledge action after applying the pgvector migration."
                )
        names = {d.id: d.filename for d in db.query(KnowledgeDoc).filter(KnowledgeDoc.exam_id == exam_id).all()}
        return cls(db, exam_id, matching, names, embedder)

    def _bm25(self, query: str) -> np.ndarray:
        q = set(tokens(query))
        scores = np.zeros(len(self.rows), dtype=np.float32)
        for i, toks in enumerate(self.docs_tokens):
            if not toks:
                continue
            length = len(toks)
            for term in q:
                freq = toks.count(term)
                if freq:
                    scores[i] += self.idf.get(term, 0.0) * freq * 2.5 / (freq + 1.5 * (1 - 0.75 + 0.75 * length / max(1, self.avg_len)))
        return scores

    def search(self, query: str, k: int = 5) -> list[Hit]:
        qvec = self.embedder.embed([query])[0]
        if self.is_postgres:
            # pgvector cosine distance operator; the database filters by exam before ranking.
            distance_expr = KnowledgeChunk.embedding_vector.cosine_distance(qvec.tolist())
            ranked = (self.db.query(KnowledgeChunk, distance_expr.label("distance"))
                      .filter(KnowledgeChunk.exam_id == self.exam_id,
                              KnowledgeChunk.embed_model == self.embedder.name)
                      .order_by(distance_expr)
                      .limit(max(k * 4, k)).all())
            candidates = [row[0] for row in ranked]
            vector_rank = {chunk.id: rank for rank, (chunk, _distance) in enumerate(ranked)}
            vector_score = {chunk.id: float(1.0 - distance) for chunk, distance in ranked}
        else:
            cosine = self.matrix @ qvec
            candidates = [self.rows[i] for i in np.argsort(-cosine)[:max(k * 4, k)]]
            vector_rank = {r.id: rank for rank, r in enumerate(candidates)}
            vector_score = {r.id: float(cosine[self.rows.index(r)]) for r in candidates}

        keyword = self._bm25(query)
        keyword_order = [i for i in np.argsort(-keyword) if keyword[i] > 0]
        keyword_rank = {self.rows[i].id: rank for rank, i in enumerate(keyword_order)}
        candidate_ids = set(vector_rank) | set(keyword_rank)
        fused = {}
        for cid in candidate_ids:
            score = 0.0
            if cid in vector_rank:
                score += 1.0 / (RRF_K + vector_rank[cid] + 1)
            if cid in keyword_rank:
                score += 1.0 / (RRF_K + keyword_rank[cid] + 1)
            fused[cid] = score
        ordered = sorted(candidate_ids, key=lambda cid: -fused[cid])[:k]
        return [Hit(cid, self.names.get(self.by_id[cid].doc_id, "?"), self.by_id[cid].page,
                    self.by_id[cid].text, vector_score.get(cid, 0.0), fused[cid]) for cid in ordered]

    def search_many(self, queries: list[str], k_total: int = 6, k_each: int = 4) -> list[Hit]:
        merged: dict[int, Hit] = {}
        for query in queries:
            for hit in self.search(query, k_each):
                if hit.chunk_id in merged:
                    old = merged[hit.chunk_id]
                    merged[hit.chunk_id] = Hit(old.chunk_id, old.source, old.page, old.text,
                                               max(old.vector_score, hit.vector_score), old.fused + hit.fused)
                else:
                    merged[hit.chunk_id] = hit
        return sorted(merged.values(), key=lambda h: -h.fused)[:k_total]
