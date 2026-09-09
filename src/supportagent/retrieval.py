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
