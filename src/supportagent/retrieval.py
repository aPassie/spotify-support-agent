"""TF-IDF nearest-neighbour retrieval over historical customer messages.

Given a new customer message, find the most similar *past* customer messages and
return them with the brand's actual replies. These exemplars ground the drafted
reply in how SpotifyCares historically handled the same issue, and their DM rate
feeds the escalation policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .textnorm import clean_for_matching


@dataclass
class Neighbor:
    idx: int
    score: float
    cust_text: str
    brand_reply: str
    reply_has_dm: bool
    reply_has_url: bool
    cust_tweet_id: int


class Retriever:
    def __init__(self, char_weight: float = 0.5):
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True, stop_words="english")
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.5, sublinear_tf=True)
        self.char_weight = char_weight
        self.X = None
        self.corpus: pd.DataFrame | None = None

    def _transform(self, texts: list[str], fit: bool = False):
        cleaned = [clean_for_matching(t).lower() for t in texts]
        if fit:
            W = self.word_vec.fit_transform(cleaned)
            C = self.char_vec.fit_transform(cleaned)
        else:
            W = self.word_vec.transform(cleaned)
            C = self.char_vec.transform(cleaned)
        X = hstack([W * (1 - self.char_weight), C * self.char_weight]).tocsr()
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel()) + 1e-9
        X = X.multiply(1 / norms[:, None]).tocsr()
        return X

    def fit(self, corpus: pd.DataFrame) -> "Retriever":
        """corpus needs columns cust_text, brand_reply, reply_has_dm, reply_has_url, cust_tweet_id."""
        self.corpus = corpus.reset_index(drop=True)
        self.X = self._transform(self.corpus.cust_text.tolist(), fit=True)
        return self

    def query(self, text: str, k: int = 5, exclude_tweet_id: int | None = None) -> list[Neighbor]:
        q = self._transform([text])
        scores = np.asarray((self.X @ q.T).todense()).ravel()
        order = np.argsort(-scores)[: k + 1]
        out = []
        for i in order:
            row = self.corpus.iloc[i]
            if exclude_tweet_id is not None and row.cust_tweet_id == exclude_tweet_id:
                continue
            out.append(
                Neighbor(int(i), float(scores[i]), row.cust_text, row.brand_reply, bool(row.reply_has_dm), bool(row.reply_has_url), int(row.cust_tweet_id))
            )
            if len(out) == k:
                break
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "Retriever":
        return joblib.load(path)


class EmbeddingRetriever:
    """Dense retrieval over past customer tweets, same interface as `Retriever`.

    Embeddings come from an OpenAI-compatible /v1/embeddings endpoint (bge-small-en-v1.5
    served by llama.cpp; see scripts/setup_llm.sh embed). TF-IDF ranks by shared vocabulary,
    which is why "iOS update broke playback" retrieves "iOS update lost my songs"; a sentence
    encoder ranks by meaning. This class exists to measure whether that difference matters,
    see REPORT.md failure mode 1 and the ablation table.
    """

    def __init__(self, base_url: str | None = None, model: str = "bge-small-en-v1.5", batch: int = 64, parallel: int = 4):
        import os

        self.base_url = (base_url or os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:8082/v1")).rstrip("/")
        self.model = model
        self.batch = batch
        self.parallel = parallel
        self.X = None
        self.corpus: pd.DataFrame | None = None

    # -- embedding ---------------------------------------------------------
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        import requests

        r = requests.post(f"{self.base_url}/embeddings", json={"model": self.model, "input": texts}, timeout=600)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        E = np.asarray([d["embedding"] for d in data], dtype=np.float32)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def embed(self, texts: list[str], desc: str = "embed") -> np.ndarray:
        import concurrent.futures as cf

        from tqdm import tqdm

        cleaned = [clean_for_matching(t) or "empty" for t in texts]
        chunks = [cleaned[i : i + self.batch] for i in range(0, len(cleaned), self.batch)]
        out: list[np.ndarray | None] = [None] * len(chunks)
        with cf.ThreadPoolExecutor(self.parallel) as ex:
            futs = {ex.submit(self._embed_batch, c): i for i, c in enumerate(chunks)}
            for f in tqdm(cf.as_completed(futs), total=len(chunks), desc=desc, mininterval=5):
                out[futs[f]] = f.result()
        return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)

    # -- same API as Retriever --------------------------------------------
    def fit(self, corpus: pd.DataFrame) -> "EmbeddingRetriever":
        self.corpus = corpus.reset_index(drop=True)
        self.X = self.embed(self.corpus.cust_text.tolist(), desc="embed-corpus")
        return self

    def query(self, text: str, k: int = 5, exclude_tweet_id: int | None = None) -> list[Neighbor]:
        q = self.embed([text], desc="embed-q")[0]
        scores = self.X @ q
        order = np.argsort(-scores)[: k + 1]
        out = []
        for i in order:
            row = self.corpus.iloc[i]
            if exclude_tweet_id is not None and row.cust_tweet_id == exclude_tweet_id:
                continue
            out.append(
                Neighbor(int(i), float(scores[i]), row.cust_text, row.brand_reply, bool(row.reply_has_dm), bool(row.reply_has_url), int(row.cust_tweet_id))
            )
            if len(out) == k:
                break
        return out

    def query_many(self, texts: list[str], k: int = 5, exclude_tweet_ids: list[int] | None = None) -> list[list[Neighbor]]:
        """Batch version: one embedding round-trip for all queries."""
        Q = self.embed(texts, desc="embed-queries")
        S = self.X @ Q.T  # (corpus, queries)
        res = []
        for j in range(S.shape[1]):
            scores = S[:, j]
            order = np.argsort(-scores)[: k + 1]
            ex = exclude_tweet_ids[j] if exclude_tweet_ids else None
            nbrs = []
            for i in order:
                row = self.corpus.iloc[i]
                if ex is not None and row.cust_tweet_id == ex:
                    continue
                nbrs.append(Neighbor(int(i), float(scores[i]), row.cust_text, row.brand_reply, bool(row.reply_has_dm), bool(row.reply_has_url), int(row.cust_tweet_id)))
                if len(nbrs) == k:
                    break
            res.append(nbrs)
        return res

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self.X)
        self.corpus.to_parquet(path.with_suffix(".corpus.parquet"))

    @staticmethod
    def load(path: Path, **kw) -> "EmbeddingRetriever":
        r = EmbeddingRetriever(**kw)
        r.X = np.load(path.with_suffix(".npy"))
        r.corpus = pd.read_parquet(path.with_suffix(".corpus.parquet"))
        return r
