"""Choose the policy thresholds on held-out HISTORICAL data (never the golden set).

For each candidate threshold we ask: of the historical messages the policy would
auto-handle, what share does the LLM teacher consider an escalate-only intent? That is the
quantity the threshold exists to control. We report the trade-off curve and pick the
loosest threshold whose leakage stays under a stated budget, so the choice is auditable.
"""
from __future__ import annotations

import argparse

import numpy as np
import yaml
from sklearn.model_selection import cross_val_predict

from supportagent.intents import TfidfClassifier, load_taxonomy, read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", default="data/processed/silver_labels.jsonl")
    ap.add_argument("--leak-budget", type=float, default=0.10, help="max share of auto-handled items whose true intent is escalate-only")
    args = ap.parse_args()
    tax = load_taxonomy()
    esc = {i.name for i in tax.intents if not i.auto_handle}
    rows = read_jsonl(args.silver)
    texts = [r["cust_text"] for r in rows]
    y = np.array([r["intent"] for r in rows])
    clf = TfidfClassifier()
    P = cross_val_predict(clf.pipe, clf._prep(texts), y, cv=5, method="predict_proba")
    classes = list(np.unique(y))
    esc_idx = [j for j, c in enumerate(classes) if c in esc]
    p_esc = P[:, esc_idx].sum(axis=1)
    top = P.max(axis=1)
    truth_esc = np.isin(y, list(esc))

    print(f"{len(rows)} historical messages; {truth_esc.mean():.1%} have an escalate-only intent (per LLM teacher)\n")
    print(" thr | auto share | leakage (auto but truly escalate) | escalate recall")
    print("-----|------------|----------------------------------|----------------")
    best = None
    for thr in np.arange(0.20, 0.81, 0.05):
        auto = p_esc < thr
        if auto.sum() == 0:
            continue
        leak = truth_esc[auto].mean()
        rec = (truth_esc & ~auto).sum() / truth_esc.sum()
        flag = ""
        if leak <= args.leak_budget:
            best = thr if best is None or thr > best else best
            flag = "  <- within budget"
        print(f"{thr:4.2f} |   {auto.mean():.3f}    |             {leak:.3f}              |     {rec:.3f}{flag}")
    print(f"\nloosest threshold with leakage <= {args.leak_budget:.0%}: {best:.2f}" if best else "\nno threshold meets the budget")

    print("\ntop-1 confidence floor (share of items and their accuracy below each floor):")
    pred = np.array(classes)[P.argmax(axis=1)]
    correct = pred == y
    for floor in (0.15, 0.18, 0.20, 0.22, 0.25, 0.30):
        low = top < floor
        if low.sum():
            print(f"  top1 < {floor:.2f}: {low.mean():.3f} of items, accuracy {correct[low].mean():.3f} (overall {correct.mean():.3f})")
    cfg = yaml.safe_load(open("configs/agent.yaml"))
    print(f"\nconfig currently: escalate_prob_threshold={cfg['policy']['escalate_prob_threshold']}, min_top_confidence={cfg['policy']['min_top_confidence']}")


if __name__ == "__main__":
    main()
