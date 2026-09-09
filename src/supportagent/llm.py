"""Thin OpenAI-compatible chat client with an on-disk cache.

Works with any OpenAI-compatible endpoint. Defaults target a local llama.cpp server
(see scripts/setup_llm.sh). Configure with env vars:
  LLM_BASE_URL (default http://127.0.0.1:8080/v1)
  LLM_API_KEY  (default "none")
  LLM_MODEL    (default "local")
Every call is cached in a sqlite db keyed by a hash of (model, messages, params), so
re-running the pipeline is free and deterministic once outputs exist.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

CACHE_PATH = Path(os.environ.get("LLM_CACHE", "outputs/cache/llm_cache.sqlite"))


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        cache_path: Path = CACHE_PATH,
        parallel: int = 4,
        offline: bool = False,
    ):
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "none")
        self.model = model or os.environ.get("LLM_MODEL", "local")
        self.parallel = parallel
        self.offline = offline  # cache-only: raise if a call is not cached
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, model TEXT, v TEXT)")
        self._db.commit()
        self._lock = threading.Lock()
        self._client = None
        self.n_calls = 0
        self.n_cache_hits = 0

    # -- cache -------------------------------------------------------------
    @staticmethod
    def _key(model: str, messages: list[dict], params: dict) -> str:
        blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _get(self, k: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def _put(self, k: str, v: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache (k, model, v) VALUES (?,?,?)", (k, self.model, v))
            self._db.commit()

    # -- calls -------------------------------------------------------------
    def _openai(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=600, max_retries=2)
        return self._client

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 200,
        temperature: float = 0.0,
        json_schema: dict | None = None,
        seed: int = 0,
        tag: str | None = None,
    ) -> str:
        """Return the assistant message content (string). Deterministic by default (temp 0, fixed seed)."""
        params: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature, "seed": seed}
        if json_schema is not None:
            params["json_schema"] = json_schema
        key = self._key(self.model, messages, params)
        cached = self._get(key)
        if cached is not None:
            self.n_cache_hits += 1
            return cached
        if self.offline:
            raise RuntimeError(f"LLM call not in cache and client is offline (tag={tag})")
        kwargs: dict[str, Any] = dict(model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature, seed=seed)
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": json_schema, "strict": True}}
        last_err = None
        for attempt in range(4):
            try:
                resp = self._openai().chat.completions.create(**kwargs)
                out = resp.choices[0].message.content or ""
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"LLM call failed after retries: {last_err}")
        self.n_calls += 1
        self._put(key, out)
        return out

    def chat_many(self, batches: Iterable[dict], desc: str = "llm") -> list[str]:
        """Run many chat() calls concurrently. Each item is a kwargs dict for chat()."""
        from tqdm import tqdm

        items = list(batches)
        out: list[str | None] = [None] * len(items)
        with cf.ThreadPoolExecutor(self.parallel) as ex:
            futs = {ex.submit(self.chat, **kw): i for i, kw in enumerate(items)}
            for f in tqdm(cf.as_completed(futs), total=len(items), desc=desc, mininterval=5):
                out[futs[f]] = f.result()
        return out  # type: ignore[return-value]


def parse_json(text: str) -> dict:
    """Parse JSON from a model response, tolerating code fences and leading prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in: {text[:200]!r}")
    return json.loads(t[start : end + 1])
