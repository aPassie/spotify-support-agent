"""Judge every system's replies on the golden set with the judge model, and create the
blinded human-rating sheet. Judge endpoint is configured separately from the generator:
  JUDGE_BASE_URL (default http://127.0.0.1:8081/v1), JUDGE_MODEL (default "judge")."""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import yaml

from .intents import read_jsonl, write_jsonl
from .judge import judge_many
from .llm import LLMClient
from .retrieval import Retriever

SYSTEMS = {"agent": "agent.jsonl", "trivial": "baseline_trivial.jsonl", "simple": "baseline_simple.jsonl"}
JUDGE = Path("outputs/judge")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="golden/golden.jsonl")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--n-human", type=int, default=20, help="items per system in the human rating sheet")
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/agent.yaml"))
    gold = {g["id"]: g for g in read_jsonl(Path(args.golden))}
    retr = Retriever.load(Path("outputs/models/retriever.joblib"))
    client = LLMClient(
        base_url=os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8081/v1"),
        model=os.environ.get("JUDGE_MODEL", "judge"),
        parallel=args.parallel,
        offline=args.offline,
    )
    similar_cache: dict[str, list[dict]] = {}
    for s, f in SYSTEMS.items():
        p = Path("outputs/predictions") / f
        if not p.exists():
            continue
        preds = read_jsonl(p)
        items = []
        for r in preds:
            g = gold[r["id"]]
            if r["id"] not in similar_cache:
                nb = retr.query(g["cust_text"], k=3, exclude_tweet_id=g["cust_tweet_id"])
                similar_cache[r["id"]] = [{"cust_text": n.cust_text, "brand_reply": n.brand_reply} for n in nb]
            items.append({"cust_text": g["cust_text"], "reply": r["reply"], "hist_reply": g["hist_brand_reply"], "similar": similar_cache[r["id"]]})
        res = judge_many(client, items, desc=f"judge:{s}")
        write_jsonl(JUDGE / f"{s}.jsonl", [{"id": r["id"], "system": s, "reply": r["reply"]} | j for r, j in zip(preds, res)])
        import numpy as np

        print(f"{s}: mean overall {np.mean([j['overall'] for j in res]):.2f}, parse failures {sum(not j['parse_ok'] for j in res)}")
    print(f"judge llm calls={client.n_calls} cache hits={client.n_cache_hits}")

    # blinded human rating sheet: same ids for every system, shuffled, system hidden in a key file
    sheet = Path("golden/human_ratings.csv")
    if not sheet.exists():
        rng = random.Random(cfg["golden"]["seed"])
        ids = sorted(gold)
        chosen = rng.sample(ids, args.n_human)
        rows = []
        for s, f in SYSTEMS.items():
            preds = {r["id"]: r for r in read_jsonl(Path("outputs/predictions") / f)}
            for i in chosen:
                rows.append({"id": i, "system": s, "cust_text": gold[i]["text_display"], "reply": preds[i]["reply"], "human_overall": ""})
        rng.shuffle(rows)
        with open(sheet, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "system", "cust_text", "reply", "human_overall"])
            w.writeheader(); w.writerows(rows)
        print(f"wrote blank human rating sheet {sheet} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
