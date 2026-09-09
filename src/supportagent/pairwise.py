"""Blinded pairwise reply comparison between two systems.

Absolute 1-5 scoring is noisy and, as the judge-vs-human numbers show, badly calibrated. For
"is variant B's reply better than variant A's?" a paired preference is the right instrument: it
holds the customer tweet fixed, needs no shared scale, and its null hypothesis is exactly 50%.

`make-sheet` writes a CSV with the two replies as A/B in randomised order and a separate key
file. `score` reads the filled sheet and reports the win rate with an exact binomial CI.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from .intents import read_jsonl

PRED = Path("outputs/predictions")


def _load(system: str) -> dict[str, dict]:
    f = "agent.jsonl" if system == "agent" else f"baseline_{system}.jsonl" if system in ("simple", "trivial") else f"ablation_{system}.jsonl"
    return {r["id"]: r for r in read_jsonl(PRED / f)}


def make_sheet(a: str, b: str, out: Path, n: int, seed: int, auto_only: bool) -> None:
    gold = {g["id"]: g for g in read_jsonl(Path("golden/golden.jsonl"))}
    A, B = _load(a), _load(b)
    ids = [
        i for i in sorted(gold)
        if i in A and i in B and A[i]["reply"] != B[i]["reply"]
        and (not auto_only or (A[i]["action"] == "auto" and B[i]["action"] == "auto"))
    ]
    rng = random.Random(seed)
    chosen = rng.sample(ids, min(n, len(ids)))
    rows, key = [], []
    for i in chosen:
        flip = rng.random() < 0.5
        left, right = (B[i], A[i]) if flip else (A[i], B[i])
        rows.append({"id": i, "cust_text": gold[i]["text_display"], "reply_A": left["reply"], "reply_B": right["reply"], "better": ""})
        key.append({"id": i, "A": b if flip else a, "B": a if flip else b})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "cust_text", "reply_A", "reply_B", "better"]); w.writeheader(); w.writerows(rows)
    json.dump({"system_a": a, "system_b": b, "auto_only": auto_only, "eligible": len(ids), "key": key}, open(out.with_suffix(".key.json"), "w"), indent=1)
    print(f"{len(ids)} items differ between {a} and {b}; wrote {len(rows)} to {out} (+ hidden key)")


def score(sheet: Path) -> dict:
    from scipy.stats import binomtest

    key = json.load(open(sheet.with_suffix(".key.json")))
    side = {k["id"]: k for k in key["key"]}
    rows = [r for r in csv.DictReader(open(sheet)) if r["better"].strip()]
    wins = {key["system_a"]: 0, key["system_b"]: 0, "tie": 0}
    for r in rows:
        v = r["better"].strip().upper()
        wins["tie" if v in ("T", "TIE", "=") else side[r["id"]][v]] += 1
    a, b = key["system_a"], key["system_b"]
    decided = wins[a] + wins[b]
    test = binomtest(wins[b], decided, 0.5) if decided else None
    out = {
        "system_a": a, "system_b": b, "n_rated": len(rows), "n_decided": decided, "ties": wins["tie"],
        f"wins_{a}": wins[a], f"wins_{b}": wins[b],
        "win_rate_b_of_decided": wins[b] / decided if decided else float("nan"),
        "ci95": list(test.proportion_ci(0.95)) if test else [float("nan")] * 2,
        "p_value": float(test.pvalue) if test else float("nan"),
    }
    # How many rated pairs would it take to call an effect this size? Reported so a
    # non-significant result is read as "underpowered", not as "no difference".
    if decided:
        out["power"] = {
            "observed_win_rate": out["win_rate_b_of_decided"],
            "n_decided_for_80pct_power": _n_for_power(out["win_rate_b_of_decided"]),
            "tie_rate": wins["tie"] / len(rows) if rows else float("nan"),
            "eligible_items": key["eligible"],
        }
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("outputs/metrics/pairwise.json", "w"), indent=1)
    return out


def _n_for_power(p: float, alpha: float = 0.05, power: float = 0.80) -> int | None:
    """Decided pairs needed for a two-sided binomial test at this effect size (normal approx)."""
    from scipy.stats import norm

    if not (0 < p < 1) or abs(p - 0.5) < 1e-9:
        return None
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return int(np.ceil(((z_a * 0.5 + z_b * np.sqrt(p * (1 - p))) / (p - 0.5)) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-sheet"); m.add_argument("--a", default="agent"); m.add_argument("--b", default="embed")
    m.add_argument("--out", default="golden/pairwise_agent_vs_embed.csv"); m.add_argument("--n", type=int, default=30)
    m.add_argument("--seed", type=int, default=0); m.add_argument("--all-items", action="store_true")
    s = sub.add_parser("score"); s.add_argument("--sheet", default="golden/pairwise_agent_vs_embed.csv")
    args = ap.parse_args()
    if args.cmd == "make-sheet":
        make_sheet(args.a, args.b, Path(args.out), args.n, args.seed, auto_only=not args.all_items)
    else:
        print(json.dumps(score(Path(args.sheet)), indent=1))


if __name__ == "__main__":
    main()
