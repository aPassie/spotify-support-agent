"""Escalation policy: decide whether a message is auto-handled or escalated, with reasons.

The policy is deliberately rule-based and inspectable: every escalation carries one or
more stable reason ids from configs/intents.yaml so that we can measure *why* messages
escalate, not just how often.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .intents import Taxonomy
from .retrieval import Neighbor
from .textnorm import clean_for_matching

LEGAL_SAFETY_RE = re.compile(
    r"\b(lawsuit|sue you|suing|lawyer|attorney|legal action|police|court|fraud|scam(?:med)?|stolen|"
    r"kill myself|suicid|self[- ]harm|harass|threat|discriminat|gdpr|data protection|press|journalist)\b",
    re.I,
)
SECURITY_RE = re.compile(r"\b(hacked|hacker|compromised|someone else is using|unknown device|not me|didn'?t authori[sz]e)\b", re.I)
MONEY_RE = re.compile(r"\b(refund|money back|charged|charging|overcharg|double charg|chargeback|dispute)\b", re.I)


@dataclass
class Decision:
    action: str  # "auto" | "escalate"
    reasons: list[str] = field(default_factory=list)

    @property
    def primary_reason(self) -> str | None:
        return self.reasons[0] if self.reasons else None

    @property
    def escalate(self) -> bool:
        return self.action == "escalate"


def decide(
    *,
    text: str,
    lang: str,
    pii: list[str],
    is_media_only: bool,
    intent: str,
    confidence: float,
    neighbors: list[Neighbor],
    tax: Taxonomy,
    cfg: dict,
    intent_probs: dict[str, float] | None = None,
) -> Decision:
    """Return the auto/escalate decision plus every reason that fired (first = primary).

    `intent_probs` is the full class distribution. The escalate gate uses the probability
    *mass* on escalate-only intents rather than top-1 confidence, because uncertainty
    between two publicly-answerable intents is harmless while uncertainty between a public
    and an account-specific intent is not.
    """
    pol = cfg["policy"]
    reasons: list[str] = []
    t = clean_for_matching(text)

    # 1. can we even read the request?
    if is_media_only:
        reasons.append("media_only")
    if lang not in ("en", "und"):
        reasons.append("language_routing")
    # 2. hard safety triggers, independent of intent
    if pii:
        reasons.append("pii_in_public")
    if LEGAL_SAFETY_RE.search(t):
        reasons.append("legal_or_safety")
    if SECURITY_RE.search(t):
        reasons.append("security_incident")
    if MONEY_RE.search(t):
        reasons.append("payment_dispute")
    # 3. intent-level policy, on probability mass rather than the argmax alone
    escalate_names = [i.name for i in tax.intents if not i.auto_handle]
    if intent_probs:
        p_esc = sum(intent_probs.get(n, 0.0) for n in escalate_names)
        # which escalate-intent carries most of that mass decides the reason we report
        worst = max(escalate_names, key=lambda n: intent_probs.get(n, 0.0))
    else:
        p_esc = 1.0 if intent in escalate_names else 0.0
        worst = intent if intent in escalate_names else escalate_names[0]
    if p_esc >= pol["escalate_prob_threshold"]:
        reasons.append("unclear_request" if worst == "other" else "needs_account_access")
    if confidence < pol["min_top_confidence"]:
        reasons.append("low_confidence")
    # 4. evidence from history: is there a precedent, and did the brand handle it publicly?
    if neighbors:
        top_sim = max(n.score for n in neighbors)
        dm_rate = sum(n.reply_has_dm for n in neighbors) / len(neighbors)
        if top_sim < cfg["retrieval"]["min_similarity"]:
            reasons.append("no_precedent")
        if dm_rate >= pol["neighbor_dm_rate_escalate"]:
            reasons.append("historical_dm_pattern")
    else:
        reasons.append("no_precedent")

    # de-duplicate, keep order (first = primary)
    seen: set[str] = set()
    reasons = [r for r in reasons if not (r in seen or seen.add(r))]
    return Decision("escalate" if reasons else "auto", reasons)
