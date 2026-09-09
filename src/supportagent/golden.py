"""Sample the golden evaluation set (to be hand-labelled) from the held-out period.

Two strata (recorded in `sample_strategy` so metrics can be reported on the unbiased
random stratum alone):
  random   - uniform sample of root customer tweets from the eval period
  targeted - keyword-matched extras for intents that are rare in a uniform sample
Writes golden/golden_candidates.jsonl with empty label fields.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from .data import load_roots
from .intents import load_taxonomy, write_jsonl
from .textnorm import clean_for_matching


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="golden/golden_candidates.jsonl")
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/agent.yaml"))
    g = cfg["golden"]
    tax = load_taxonomy()
    roots = load_roots(cfg["brand"])
    pool = roots[(roots.split == "eval") & roots.cust_text.notna()].copy()
    rng_seed = g["seed"]
    rand = pool.sample(n=g["n_random"], random_state=rng_seed)
    rand["sample_strategy"] = "random"
    rest = pool.drop(rand.index)
    parts = [rand]
    for name in g["targeted_intents"]:
        kws = tax.get(name).seed_keywords
        pat = re.compile("|".join(r"(?<![a-z])" + re.escape(str(k).lower()) + r"(?![a-z])" for k in kws))
        m = rest[rest.cust_text.map(lambda t: bool(pat.search(clean_for_matching(t).lower()))) & (rest.lang == "en")]
        pick = m.sample(n=min(g["n_targeted_per_intent"], len(m)), random_state=rng_seed)
        pick["sample_strategy"] = f"targeted:{name}"
        rest = rest.drop(pick.index)
        parts.append(pick)
    gold = pd.concat(parts).sort_values("cust_created_at").reset_index(drop=True)
    rows = []
    for i, r in gold.iterrows():
        rows.append(
            {
                "id": f"g{i:03d}",
                "cust_tweet_id": int(r.cust_tweet_id),
                "cust_created_at": str(r.cust_created_at),
                "cust_text": r.cust_text,
                "text_display": r.text_display,
                "lang": r.lang,
                "pii": r.pii,
                "sample_strategy": r.sample_strategy,
                "hist_brand_reply": r.brand_reply,          # what SpotifyCares actually replied (reference, not label)
                "hist_reply_has_dm": bool(r.reply_has_dm),
                # ---- to be filled by hand ----
                "intent": None,
                "should_escalate": None,
                "escalation_reason": None,
                "notes": None,
            }
        )
    write_jsonl(Path(args.out), rows)
    print(f"wrote {len(rows)} candidates to {args.out}")
    print(gold.sample_strategy.value_counts().to_string())
    print("languages:", gold.lang.value_counts().head(6).to_dict())


if __name__ == "__main__":
    main()
