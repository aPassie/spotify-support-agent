"""Fit the retrieval index on historical messages and train the intent classifier on silver labels."""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import yaml
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

from .data import load_roots
from .intents import TfidfClassifier, read_jsonl
from .retrieval import Retriever
from .textnorm import extract_urls

MODELS = Path("outputs/models")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", default="data/processed/silver_labels.jsonl")
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/agent.yaml"))
    roots = load_roots(cfg["brand"])
    hist = roots[(roots.split == "history") & (~roots.is_media_only)]
    retr = Retriever().fit(hist[["cust_text", "brand_reply", "reply_has_dm", "reply_has_url", "cust_tweet_id"]])
    retr.save(MODELS / "retriever.joblib")
    print(f"retriever: {len(hist)} historical messages indexed")

    # Export the set of links SpotifyCares itself used, so the reply guardrail check in
    # evaluate.py can run without the (large, regenerable) parquet files.
    import json

    urls = sorted({u.rstrip(".,)!") for t in hist.brand_reply_all for u in extract_urls(t)})
    json.dump(urls, open(MODELS / "hist_urls.json", "w"))
    print(f"exported {len(urls)} historical links to {MODELS / 'hist_urls.json'}")

    silver = read_jsonl(Path(args.silver))
    texts = [r["cust_text"] for r in silver]
    labels = [r["intent"] for r in silver]
    print("silver label distribution:", collections.Counter(labels).most_common())
    clf = TfidfClassifier()
    # 5-fold CV on silver labels = agreement with the LLM teacher, NOT accuracy vs. truth
    pred = cross_val_predict(clf.pipe, clf._prep(texts), labels, cv=5)
    print("5-fold CV vs silver labels (teacher agreement):")
    print(classification_report(labels, pred, digits=3, zero_division=0))
    clf.fit(texts, labels)
    clf.save(MODELS / "intent_clf.joblib")
    print(f"saved {MODELS / 'intent_clf.joblib'}")


if __name__ == "__main__":
    main()
