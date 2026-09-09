"""Produce silver intent labels on historical messages with the LLM (teacher) so a cheap
classifier can be trained. Output: data/processed/silver_labels.jsonl"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .data import load_roots
from .intents import llm_label, load_taxonomy, write_jsonl
from .llm import LLMClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default="data/processed/silver_labels.jsonl")
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/agent.yaml"))
    n = args.n or cfg["silver"]["n_examples"]
    roots = load_roots(cfg["brand"])
    pool = roots[(roots.split == "history") & (roots.lang == "en") & (~roots.is_media_only)]
    sample = pool.sample(n=min(n, len(pool)), random_state=cfg["silver"]["seed"])
    tax = load_taxonomy()
    client = LLMClient(parallel=args.parallel)
    labels = llm_label(client, tax, sample.cust_text.tolist(), desc="silver")
    rows = [
        {"cust_tweet_id": int(r.cust_tweet_id), "cust_text": r.cust_text, "intent": lab, "reply_has_dm": bool(r.reply_has_dm)}
        for r, lab in zip(sample.itertuples(), labels)
    ]
    write_jsonl(Path(args.out), rows)
    import collections

    print(collections.Counter(labels).most_common())
    print(f"wrote {len(rows)} silver labels to {args.out}; llm calls={client.n_calls} cache hits={client.n_cache_hits}")


if __name__ == "__main__":
    main()
