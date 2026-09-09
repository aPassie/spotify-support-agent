"""Compute all metrics from the golden labels, the prediction files and (if present) the
judge outputs and human ratings. Writes outputs/metrics/metrics.json and tables.md.
No LLM is needed: this is the step `make reproduce` runs."""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .intents import load_taxonomy, read_jsonl
from .judge import CHECKS, agreement
from .textnorm import extract_urls

PRED = Path("outputs/predictions")
JUDGE = Path("outputs/judge")
METRICS = Path("outputs/metrics")
SYSTEMS = {"agent": "agent.jsonl", "trivial": "baseline_trivial.jsonl", "simple": "baseline_simple.jsonl"}


def all_systems() -> dict[str, str]:
    """Baselines and agent, plus any ablation_*.jsonl that has been produced."""
    out = dict(SYSTEMS)
    for f in sorted(PRED.glob("ablation_*.jsonl")):
        out[f.stem.replace("ablation_", "abl:")] = f.name
    return out


def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI over per-item values. Returns (lo, hi)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = stat(v[idx], axis=1)
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(c - h), float(c + h))


def mcnemar(a_wrong, b_wrong):
    """Exact McNemar test on paired binary outcomes (same items, two systems).

    a_wrong/b_wrong are boolean arrays of 'this system got this item wrong'. Returns the
    discordant counts and a two-sided exact binomial p-value.
    """
    from scipy.stats import binomtest

    a_wrong = np.asarray(a_wrong, dtype=bool)
    b_wrong = np.asarray(b_wrong, dtype=bool)
    n01 = int((~a_wrong & b_wrong).sum())  # only b wrong
    n10 = int((a_wrong & ~b_wrong).sum())  # only a wrong
    n = n01 + n10
    p = float(binomtest(n10, n, 0.5).pvalue) if n else 1.0
    return {"only_b_wrong": n01, "only_a_wrong": n10, "discordant": n, "p_value": p}


def _by_id(rows):
    return {r["id"]: r for r in rows}


def intent_metrics(gold, pred, names):
    y = [g["intent"] for g in gold]
    p = [pred[g["id"]]["intent"] for g in gold]
    per_class = precision_recall_fscore_support(y, p, labels=names, zero_division=0)
    correct = [int(a == b) for a, b in zip(y, p)]
    return {
        "accuracy": accuracy_score(y, p),
        "accuracy_ci": bootstrap_ci(correct),
        "macro_f1": f1_score(y, p, average="macro", labels=names, zero_division=0),
        "per_class_f1": dict(zip(names, per_class[2].tolist())),
        "support": dict(zip(names, per_class[3].tolist())),
    }


def escalation_metrics(gold, pred):
    y = np.array([g["should_escalate"] for g in gold])
    p = np.array([pred[g["id"]]["action"] == "escalate" for g in gold])
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum()); tn = int((~y & ~p).sum())
    return {
        "coverage_auto_rate": float((~p).mean()),
        "coverage_auto_rate_ci": bootstrap_ci((~p).astype(float)),
        "escalate_precision": tp / (tp + fp) if tp + fp else 0.0,
        "escalate_recall": tp / (tp + fn) if tp + fn else 0.0,
        "unsafe_auto_rate": fn / (tp + fn) if tp + fn else 0.0,  # should escalate but auto-handled
        "auto_precision": tn / (tn + fn) if tn + fn else 0.0,      # auto-handled and that was right
        "auto_precision_ci": wilson_ci(tn, tn + fn),
        "n_auto": tn + fn,
        "n_unsafe_auto": fn,
        "n_over_escalated": fp,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        # per-item outcome for paired significance tests: did this system send a public reply
        # to an item that needed a human?
        "unsafe_per_item": (y & ~p).astype(int).tolist(),
        "wrong_decision_per_item": ((y & ~p) | (~y & p)).astype(int).tolist(),
    }


NAME_RE = re.compile(r"\b(?:Hey|Hi|Hello)\s+([A-Z][a-z]{2,})\b")


def reply_auto_metrics(gold, pred, hist_urls):
    """Surface checks that need no model: length, link provenance, wrong-name usage, overlap.

    `pct_wrong_name` counts replies that greet the customer by a first name. The dataset is
    anonymised, so no reply can legitimately know a customer's name: any name is copied from a
    different customer's historical reply and is therefore wrong.
    """
    lens, has_url, bad_url, rouge, named = [], [], [], [], []
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for g in gold:
        r = pred[g["id"]]["reply"]
        lens.append(len(r))
        urls = extract_urls(r)
        has_url.append(bool(urls))
        bad_url.append(any(u.rstrip(".,)!") not in hist_urls for u in urls))
        named.append(bool(NAME_RE.search(r)))
        rouge.append(sc.score(g["hist_brand_reply"], r)["rougeL"].fmeasure)
    return {"mean_len": float(np.mean(lens)), "pct_over_280": float(np.mean([l > 280 for l in lens])), "pct_with_url": float(np.mean(has_url)), "pct_unknown_url": float(np.mean(bad_url)), "pct_wrong_name": float(np.mean(named)), "rougeL_vs_hist": float(np.mean(rouge))}


def judge_metrics(gold, pred, jrows):
    j = _by_id(jrows)
    ids = [g["id"] for g in gold if g["id"] in j]
    if not ids:
        return None
    ov = np.array([j[i]["overall"] for i in ids])
    out = {"n": len(ids), "mean_overall": float(ov.mean()), "mean_overall_ci": bootstrap_ci(ov), "pct_ge4": float((ov >= 4).mean()), "pct_le2": float((ov <= 2).mean())}
    for c in CHECKS:
        out[f"pass_{c}"] = float(np.mean([bool(j[i].get(c)) for i in ids]))
    gd = _by_id(gold)
    for name, mask in [("label_auto_ok", lambda i: not gd[i]["should_escalate"]), ("label_escalate", lambda i: gd[i]["should_escalate"]),
                       ("system_auto", lambda i: pred[i]["action"] == "auto"), ("system_escalate", lambda i: pred[i]["action"] == "escalate")]:
        sub = [i for i in ids if mask(i)]
        if sub:
            s = np.array([j[i]["overall"] for i in sub])
            out[f"{name}_n"] = len(sub); out[f"{name}_mean"] = float(s.mean()); out[f"{name}_pct_ge4"] = float((s >= 4).mean())
    return out


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="golden/golden.jsonl")
    args = ap.parse_args()
    tax = load_taxonomy()
    names = tax.names
    gold_all = read_jsonl(Path(args.golden))
    systems = all_systems()
    preds = {s: _by_id(read_jsonl(PRED / f)) for s, f in systems.items() if (PRED / f).exists()}
    judges = {s: read_jsonl(JUDGE / f"{s.replace(':', '_')}.jsonl") for s in systems if (JUDGE / f"{s.replace(':', '_')}.jsonl").exists()}
    # Link allowlist: prefer the small exported file so this step needs no large data files.
    hist_urls: set[str] = set()
    url_file = Path("outputs/models/hist_urls.json")
    if url_file.exists():
        hist_urls = set(json.load(open(url_file)))
    elif Path("data/processed/SpotifyCares_roots.parquet").exists():
        import pandas as pd

        roots = pd.read_parquet("data/processed/SpotifyCares_roots.parquet")
        for t in roots[roots.split == "history"].brand_reply_all:
            hist_urls |= {u.rstrip(".,)!") for u in extract_urls(t)}

    strata = {"random": [g for g in gold_all if g["sample_strategy"] == "random"], "all": gold_all}
    M: dict = {"n_golden": len(gold_all), "n_random": len(strata["random"]), "systems": {}}
    for s, pred in preds.items():
        M["systems"][s] = {}
        for st, gold in strata.items():
            gold = [g for g in gold if g["id"] in pred]
            M["systems"][s][st] = {
                "intent": intent_metrics(gold, pred, names),
                "escalation": escalation_metrics(gold, pred),
                "reply_auto": reply_auto_metrics(gold, pred, hist_urls) if hist_urls else None,
                "judge": judge_metrics(gold, pred, judges[s]) if s in judges else None,
            }
    # paired significance tests on the random stratum: is the agent's advantage real?
    if "agent" in preds:
        gold_r = [g for g in strata["random"] if g["id"] in preds["agent"]]
        M["significance_random"] = {}
        base = M["systems"]["agent"]["random"]["escalation"]
        for other in [s for s in preds if s != "agent"]:
            o = M["systems"][other]["random"]["escalation"]
            M["significance_random"][f"agent_vs_{other}"] = {
                "unsafe_auto_mcnemar": mcnemar(base["unsafe_per_item"], o["unsafe_per_item"]),
                "wrong_decision_mcnemar": mcnemar(base["wrong_decision_per_item"], o["wrong_decision_per_item"]),
                "intent_mcnemar": mcnemar(
                    [int(preds["agent"][g["id"]]["intent"] != g["intent"]) for g in gold_r],
                    [int(preds[other][g["id"]]["intent"] != g["intent"]) for g in gold_r],
                ),
            }

    # agent confusion matrix + escalation reasons
    if "agent" in preds:
        gold = gold_all
        y = [g["intent"] for g in gold]; p = [preds["agent"][g["id"]]["intent"] for g in gold]
        cm = confusion_matrix(y, p, labels=names)
        M["agent_confusion"] = {"labels": names, "matrix": cm.tolist()}
        M["agent_reasons"] = collections.Counter(preds["agent"][g["id"]]["reasons"][0] for g in gold if preds["agent"][g["id"]]["reasons"]).most_common()
        M["agent_guardrails"] = collections.Counter(f for g in gold for f in preds["agent"][g["id"]].get("guardrail_flags", [])).most_common()
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8); ax.set_yticklabels(names, fontsize=8)
            for i in range(len(names)):
                for j in range(len(names)):
                    if cm[i, j]:
                        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xlabel("predicted"); ax.set_ylabel("hand label"); ax.set_title("Agent intent confusion (all 200)")
            fig.tight_layout(); (METRICS / "figures").mkdir(parents=True, exist_ok=True); fig.savefig(METRICS / "figures" / "confusion_agent.png", dpi=130)
        except Exception as e:  # noqa: BLE001
            print("figure skipped:", e)
    # judge vs human
    hr = Path("golden/human_ratings.csv")
    if hr.exists() and judges:
        import csv

        rows = [r for r in csv.DictReader(open(hr)) if r.get("human_overall", "").strip()]
        key = {(r["id"], r["system"]): int(r["human_overall"]) for r in rows}
        js, hs, per = [], [], collections.defaultdict(lambda: ([], []))
        for (i, s), h in key.items():
            j = _by_id(judges.get(s, []))
            if i in j:
                js.append(j[i]["overall"]); hs.append(h); per[s][0].append(j[i]["overall"]); per[s][1].append(h)
        if js:
            M["judge_vs_human"] = {"all": agreement(js, hs)} | {s: agreement(*v) for s, v in per.items()}
            # does the judge rank systems the same way as the human?
            M["judge_vs_human"]["system_means"] = {s: {"judge": float(np.mean(v[0])), "human": float(np.mean(v[1])), "n": len(v[0])} for s, v in per.items()}
    METRICS.mkdir(parents=True, exist_ok=True)
    json.dump(M, open(METRICS / "metrics.json", "w"), indent=1, default=str)

    # ---- markdown tables
    L = [f"# Metrics (golden n={M['n_golden']}, random stratum n={M['n_random']})\n"]
    for st in ("random", "all"):
        L.append(f"\n## Intent classification ({st})\n\n| system | accuracy [95% CI] | macro-F1 |\n|---|---|---|")
        for s in preds:
            m = M["systems"][s][st]["intent"]
            ci = m["accuracy_ci"]
            L.append(f"| {s} | {m['accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {m['macro_f1']:.3f} |")
    L.append(
        "\n## Escalation decision, random stratum with 95% CIs\n\n"
        "`auto-handle rate` = share of tweets answered without a human. `auto precision` = of those, "
        "the share that should indeed have been answered publicly. `unsafe replies` = public replies sent "
        "on tweets that needed a human (the error that matters).\n\n"
        "| system | auto-handle rate [CI] | auto precision [CI] | unsafe replies | escalate recall |\n|---|---|---|---|---|"
    )
    for s in preds:
        e = M["systems"][s]["random"]["escalation"]
        cc, pc = e["coverage_auto_rate_ci"], e["auto_precision_ci"]
        ap = f"{e['auto_precision']:.3f} [{pc[0]:.3f}, {pc[1]:.3f}]" if e["n_auto"] else "n/a"
        L.append(
            f"| {s} | {e['coverage_auto_rate']:.3f} [{cc[0]:.3f}, {cc[1]:.3f}] | {ap} | "
            f"{e['n_unsafe_auto']} of {e['n_auto']} | {e['escalate_recall']:.3f} |"
        )
    if "significance_random" in M:
        L.append(
            "\n### Paired significance vs the agent (McNemar exact, random stratum)\n\n"
            "Same 160 items for every system, so the comparison is paired. `discordant` counts items where "
            "exactly one of the two systems erred.\n\n"
            "| comparison | agent-only errors | other-only errors | discordant | p |\n|---|---|---|---|---|"
        )
        for k, v in M["significance_random"].items():
            for metric, label in (("unsafe_auto_mcnemar", "unsafe replies"), ("intent_mcnemar", "intent errors")):
                t = v[metric]
                L.append(f"| {k.replace('agent_vs_', 'vs ')} ({label}) | {t['only_a_wrong']} | {t['only_b_wrong']} | {t['discordant']} | {t['p_value']:.2g} |")
    L.append("\n## Escalation decision (all 200)\n\n| system | auto-handle rate | auto precision | unsafe auto rate | escalate precision | escalate recall |\n|---|---|---|---|---|---|")
    for s in preds:
        e = M["systems"][s]["all"]["escalation"]
        L.append(f"| {s} | {e['coverage_auto_rate']:.3f} | {e['auto_precision']:.3f} | {e['unsafe_auto_rate']:.3f} ({e['n_unsafe_auto']}) | {e['escalate_precision']:.3f} | {e['escalate_recall']:.3f} |")
    if any(M["systems"][s]["all"]["judge"] for s in preds):
        L.append("\n## Reply quality, LLM judge (all 200)\n\n| system | mean overall [95% CI] | % >=4 | % <=2 | addresses | brand-consistent | actionable | safe | tone |\n|---|---|---|---|---|---|---|---|---|")
        for s in preds:
            j = M["systems"][s]["all"]["judge"]
            if j:
                ci = j["mean_overall_ci"]
                L.append(f"| {s} | {j['mean_overall']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {j['pct_ge4']:.2f} | {j['pct_le2']:.2f} | " + " | ".join(f"{j['pass_'+c]:.2f}" for c in CHECKS) + " |")
        L.append("\n### Judge score by hand label (all 200)\n\n| system | label=auto-ok mean (n) | label=escalate mean (n) | system auto mean (n) | system escalate mean (n) |\n|---|---|---|---|---|")
        for s in preds:
            j = M["systems"][s]["all"]["judge"]
            if j:
                L.append(f"| {s} | {j.get('label_auto_ok_mean', float('nan')):.2f} ({j.get('label_auto_ok_n', 0)}) | {j.get('label_escalate_mean', float('nan')):.2f} ({j.get('label_escalate_n', 0)}) | {j.get('system_auto_mean', float('nan')):.2f} ({j.get('system_auto_n', 0)}) | {j.get('system_escalate_mean', float('nan')):.2f} ({j.get('system_escalate_n', 0)}) |")
    if hist_urls:
        L.append("\n## Reply surface checks (all 200)\n\n| system | mean chars | % >280 | % with link | % link not in history | % greet customer by a (necessarily wrong) name | ROUGE-L vs actual reply |\n|---|---|---|---|---|---|---|")
        for s in preds:
            a = M["systems"][s]["all"]["reply_auto"]
            L.append(f"| {s} | {a['mean_len']:.0f} | {a['pct_over_280']:.2f} | {a['pct_with_url']:.2f} | {a['pct_unknown_url']:.2f} | {a['pct_wrong_name']:.2f} | {a['rougeL_vs_hist']:.3f} |")
    if "judge_vs_human" in M:
        L.append("\n## Judge vs human agreement\n\n| subset | n | Spearman | weighted kappa | exact | within 1 | judge mean | human mean |\n|---|---|---|---|---|---|---|---|")
        for k, a in M["judge_vs_human"].items():
            if k != "system_means":
                L.append(f"| {k} | {a['n']} | {a['spearman']:.2f} | {a['kappa_quadratic']:.2f} | {a['exact_agreement']:.2f} | {a['within_one']:.2f} | {a['judge_mean']:.2f} | {a['human_mean']:.2f} |")
    if "agent" in preds:
        L.append("\n## Agent: per-class F1 (all 200)\n\n| intent | F1 | n |\n|---|---|---|")
        m = M["systems"]["agent"]["all"]["intent"]
        for n in names:
            L.append(f"| {n} | {m['per_class_f1'][n]:.2f} | {m['support'][n]} |")
        L.append("\n## Agent: primary escalation reasons\n\n| reason | n |\n|---|---|")
        for r, c in M["agent_reasons"]:
            L.append(f"| {r} | {c} |")
        L.append("\n## Agent: guardrail flags on drafts\n\n| flag | n |\n|---|---|")
        for r, c in M["agent_guardrails"]:
            L.append(f"| {r} | {c} |")
    open(METRICS / "tables.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    # failure dump for the report
    if "agent" in preds:
        fails = []
        j = _by_id(judges.get("agent", []))
        for g in gold_all:
            a = preds["agent"][g["id"]]
            why = []
            if a["intent"] != g["intent"]:
                why.append(f"intent {a['intent']}!={g['intent']}")
            if g["should_escalate"] and a["action"] == "auto":
                why.append("UNSAFE_AUTO")
            if not g["should_escalate"] and a["action"] == "escalate":
                why.append(f"over-escalated:{a['reasons'][0] if a['reasons'] else ''}")
            if g["id"] in j and j[g["id"]]["overall"] <= 2:
                why.append(f"judge={j[g['id']]['overall']}:{j[g['id']].get('rationale','')}")
            if why:
                fails.append({"id": g["id"], "why": why, "text": g["text_display"], "gold": (g["intent"], g["should_escalate"], g["escalation_reason"]), "agent": (a["intent"], round(a["confidence"], 2), a["action"], a["reasons"]), "reply": a["reply"], "hist": g["hist_brand_reply"]})
        with open(METRICS / "agent_failures.jsonl", "w") as f:
            for r in fails:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n{len(fails)} agent items with at least one problem -> {METRICS / 'agent_failures.jsonl'}")


if __name__ == "__main__":
    main()
