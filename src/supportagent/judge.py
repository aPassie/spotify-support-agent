"""LLM-as-judge for reply quality, plus judge-vs-human agreement statistics.

The judge sees the customer tweet, the candidate reply, and *reference material*: how
SpotifyCares actually replied to this tweet and to the three most similar past tweets. It
returns five binary checks, an overall 1-5 score and a one-line rationale as constrained
JSON. The judge model should be from a different family than the generator (we use
gemma-3-4b-it to judge Qwen3-4B drafts) to limit self-preference bias.
"""
from __future__ import annotations

from .llm import LLMClient, parse_json
from .baselines import clean_brand_reply
from .textnorm import display_text

JUDGE_SYSTEM = """You are a strict quality reviewer for SpotifyCares, Spotify's customer support account on Twitter. You will see a customer tweet, a candidate public reply, and reference material showing how SpotifyCares actually handled this and similar tweets.

Judge the candidate reply on these checks (true/false):
- addresses_issue: it responds to the specific problem or request in the tweet (not a generic brush-off).
- consistent_with_brand_practice: the next step it offers (troubleshooting step, help link, clarifying question, or asking to DM account email/username) is the kind of step SpotifyCares takes for this kind of issue, as shown in the references.
- actionable: the customer knows exactly what to do next after reading it.
- safe: it does not ask for passwords or payment details in public, does not invent links, policies, dates or promises (refunds, fixes), and does not contain anything embarrassing for the brand.
- on_brand_tone: friendly, concise, natural, no fake agent initials, fits in a tweet.

Then give overall: 1 = harmful or useless, 2 = poor, 3 = acceptable but generic or partly off, 4 = good, 5 = as good as the best SpotifyCares replies.
A generic "please DM us" reply to an issue that has a known public fix is at most 3. A reply that gives a public fix when the issue clearly needs account access is at most 3. Be strict: 5 is rare.

Answer with JSON only."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "addresses_issue": {"type": "boolean"},
        "consistent_with_brand_practice": {"type": "boolean"},
        "actionable": {"type": "boolean"},
        "safe": {"type": "boolean"},
        "on_brand_tone": {"type": "boolean"},
        "overall": {"type": "integer", "minimum": 1, "maximum": 5},
        "rationale": {"type": "string", "maxLength": 240},
    },
    "required": ["addresses_issue", "consistent_with_brand_practice", "actionable", "safe", "on_brand_tone", "overall", "rationale"],
    "additionalProperties": False,
}
CHECKS = ["addresses_issue", "consistent_with_brand_practice", "actionable", "safe", "on_brand_tone"]


def judge_prompt(cust_text: str, reply: str, hist_reply: str, similar: list[dict]) -> list[dict]:
    refs = "\n".join(
        f"- Similar past tweet: {display_text(s['cust_text'])}\n  SpotifyCares replied: {clean_brand_reply(s['brand_reply'])}" for s in similar[:3]
    )
    user = (
        f"Customer tweet: {display_text(cust_text)}\n\n"
        f"Reference - what SpotifyCares actually replied to this tweet: {clean_brand_reply(hist_reply)}\n\n"
        f"Reference - similar past cases:\n{refs}\n\n"
        f"Candidate reply to judge: {reply}\n\n"
        "Return the JSON."
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def judge_many(client: LLMClient, items: list[dict], desc: str = "judge") -> list[dict]:
    """items: dicts with cust_text, reply, hist_reply, similar (list of {cust_text, brand_reply})."""
    batches = [
        dict(messages=judge_prompt(it["cust_text"], it["reply"], it["hist_reply"], it["similar"]), max_tokens=160, temperature=0.0, json_schema=JUDGE_SCHEMA, tag="judge")
        for it in items
    ]
    outs = client.chat_many(batches, desc=desc)
    res = []
    for o in outs:
        try:
            d = parse_json(o)
            d["overall"] = int(max(1, min(5, d.get("overall", 3))))
            d["parse_ok"] = True
        except Exception:  # noqa: BLE001
            d = {c: False for c in CHECKS} | {"overall": 3, "rationale": "unparseable", "parse_ok": False}
        res.append(d)
    return res


def agreement(judge_scores: list[int], human_scores: list[int]) -> dict:
    """Spearman rho, quadratic-weighted Cohen's kappa, exact and within-1 agreement."""
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score

    j, h = np.asarray(judge_scores), np.asarray(human_scores)
    rho = spearmanr(j, h).statistic if len(set(j)) > 1 and len(set(h)) > 1 else float("nan")
    return {
        "n": int(len(j)),
        "spearman": float(rho),
        "kappa_quadratic": float(cohen_kappa_score(h, j, weights="quadratic")),
        "exact_agreement": float((j == h).mean()),
        "within_one": float((abs(j - h) <= 1).mean()),
        "judge_mean": float(j.mean()),
        "human_mean": float(h.mean()),
        "judge_minus_human_mean": float((j - h).mean()),
    }
