"""The support agent: classify -> retrieve precedents -> decide -> draft (with guardrails)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .baselines import DM_LINK, TRIVIAL_REPLY, clean_brand_reply
from .intents import Taxonomy, TfidfClassifier
from .llm import LLMClient
from .policy import Decision, decide
from .retrieval import Neighbor, Retriever
from .textnorm import clean_for_matching, detect_language, display_text, extract_urls, pii_flags

PROMISE_RE = re.compile(
    r"\b(we(?:'ll| will) (?:refund|credit|reimburse|fix|compensate|restore|cancel)|"
    r"you(?:'ll| will) (?:get|receive) (?:a |your )?(?:refund|credit)|full refund|refund(?:ed)? (?:you|your)|"
    r"(?:issue|process) (?:a |the )?refund|within \d+ (?:hours|days)|guarantee)\b",
    re.I,
)
SENSITIVE_ASK_RE = re.compile(r"\b(password|card number|cvv|security code|bank details|credit card details)\b", re.I)
INITIALS_RE = re.compile(r"\s*/[A-Z]{1,3}\b")
HANDLE_RE = re.compile(r"@\w+")

HANDOFF_TEMPLATES = {
    "low_confidence": "Hey! Can you tell us a bit more about what's happening? We'll see what we can do to help.",
    "no_precedent": "Hey! Can you tell us a bit more about what's happening? We'll see what we can do to help.",
    "pii_in_public": f"Hey! We'd advise deleting that tweet as it contains personal info. Could you send it to us in a DM instead? We'll take a look backstage {DM_LINK}",
    "language_routing": f"Hey! We can help out in English here on Twitter. Otherwise, send us a DM and we'll point you to the team for your language {DM_LINK}",
    "unclear_request": "Hey! Can you tell us a bit more about what's happening? We'll see what we can do to help.",
    "media_only": "Hey! We can't quite see the details there. Can you tell us a bit more about what's happening?",
    "default": TRIVIAL_REPLY,
}

SYSTEM_PROMPT = """You draft public Twitter replies for SpotifyCares, Spotify's customer support account.

Style (learned from 26,000 real SpotifyCares replies): open with a short friendly greeting ("Hey!" or "Hi there!"), be warm and concise, one to three short sentences, at most 280 characters. Sound like the example replies.

Rules:
- Ground the reply in the example replies: give the same kind of next step SpotifyCares gave for similar issues (a troubleshooting step, a help link, a clarifying question, or asking to DM the account email/username).
- Only include a link if it appears in the examples, copied exactly. Never invent links.
- Never ask for passwords or payment details. Never promise refunds, credits, fixes or dates.
- Do not sign with initials, do not include @handles, do not use hashtags.
- Output only the reply text."""


@dataclass
class AgentOutput:
    intent: str
    confidence: float
    action: str
    reasons: list[str]
    reply: str
    llm_draft: str
    guardrail_flags: list[str]
    internal_note: str
    lang: str
    pii: list[str]
    neighbor_ids: list[int]
    neighbor_scores: list[float]
    neighbor_dm_rate: float
    escalate_prob: float = 0.0
    exemplars: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def apply_guardrails(draft: str, allowed_urls: set[str], max_chars: int) -> tuple[str, list[str]]:
    flags: list[str] = []
    t = draft.strip().strip('"').strip()
    t = re.sub(r"^(reply|spotifycares)\s*:\s*", "", t, flags=re.I)
    if HANDLE_RE.search(t):
        t = HANDLE_RE.sub("", t)
        flags.append("removed_handle")
    if INITIALS_RE.search(t + " "):
        t = INITIALS_RE.sub("", t)
        flags.append("removed_initials")
    for u in extract_urls(t):
        if u.rstrip(".,)!") not in allowed_urls:
            t = t.replace(u, "")
            flags.append("hallucinated_url")
    if PROMISE_RE.search(t):
        flags.append("promise")
    if SENSITIVE_ASK_RE.search(t):
        flags.append("sensitive_ask")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_chars:
        cut = t[:max_chars]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        t = cut[: end + 1] if end > max_chars // 2 else cut.rsplit(" ", 1)[0]
        flags.append("too_long")
    if len(t) < 15:
        flags.append("empty")
    return t, flags


def format_exemplars(exemplars: list[Neighbor]) -> str:
    lines = []
    for i, n in enumerate(exemplars, 1):
        lines.append(f"{i}. Customer: {display_text(n.cust_text)}\n   SpotifyCares: {clean_brand_reply(n.brand_reply)}")
    return "\n".join(lines)


class Agent:
    def __init__(self, tax: Taxonomy, cfg: dict, retriever: Retriever, classifier: TfidfClassifier, client: LLMClient | None, k: int = 8, n_exemplars: int = 4):
        self.tax, self.cfg, self.retriever, self.classifier, self.client = tax, cfg, retriever, classifier, client
        self.k, self.n_exemplars = k, n_exemplars

    def _pick_exemplars(self, neighbors: list[Neighbor], prefer_public: bool) -> list[Neighbor]:
        if prefer_public:
            pub = [n for n in neighbors if not n.reply_has_dm]
            rest = [n for n in neighbors if n.reply_has_dm]
            ordered = pub + rest
        else:
            ordered = neighbors
        return ordered[: self.n_exemplars]

    def draft(self, text: str, intent: str, exemplars: list[Neighbor]) -> str:
        if self.client is None:
            return ""
        user = (
            f"Customer's issue category: {intent}\n\n"
            f"Similar past cases and how SpotifyCares replied:\n{format_exemplars(exemplars)}\n\n"
            f"New customer tweet: {display_text(text)}\n\nReply:"
        )
        return self.client.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            max_tokens=120,
            temperature=0.0,
            tag="draft",
        )

    def handle(self, text: str, lang: str | None = None, tweet_id: int | None = None) -> AgentOutput:
        lang = lang or detect_language(text)
        pii = pii_flags(text)
        clean = clean_for_matching(text)
        is_media_only = len(clean) < 3 and bool(extract_urls(text))
        P, classes = self.classifier.predict_proba([text])
        probs = {c: float(P[0, j]) for j, c in enumerate(classes)}
        intent = max(probs, key=probs.get)
        conf = probs[intent]
        neighbors = self.retriever.query(text, k=self.k, exclude_tweet_id=tweet_id)
        top5 = neighbors[: self.cfg["retrieval"]["k"]]
        decision: Decision = decide(
            text=text, lang=lang, pii=pii, is_media_only=is_media_only, intent=intent, confidence=conf,
            neighbors=top5, tax=self.tax, cfg=self.cfg, intent_probs=probs,
        )
        exemplars = self._pick_exemplars(neighbors, prefer_public=not decision.escalate)
        allowed_urls = {u.rstrip(".,)!") for n in exemplars for u in extract_urls(n.brand_reply)} | {DM_LINK}
        raw_draft = self.draft(text, intent, exemplars) if not (is_media_only or lang not in ("en", "und")) else ""
        draft, flags = apply_guardrails(raw_draft, allowed_urls, self.cfg["policy"]["max_reply_chars"]) if raw_draft else ("", ["no_draft"])
        action, reasons = decision.action, list(decision.reasons)
        if action == "auto" and ({"promise", "sensitive_ask", "empty", "no_draft"} & set(flags)):
            action, reasons = "escalate", ["guardrail"] + reasons
        if action == "auto":
            reply = draft
        else:
            reply = HANDOFF_TEMPLATES.get(reasons[0], HANDOFF_TEMPLATES["default"])
        dm_rate = sum(n.reply_has_dm for n in top5) / len(top5) if top5 else 1.0
        p_esc = sum(v for k, v in probs.items() if k not in self.tax.auto_handle_set)
        note = (
            f"intent={intent} ({conf:.2f}); action={action}; reasons={','.join(reasons) or '-'}; "
            f"similar past cases went to DM {dm_rate:.0%} of the time; suggested draft: {draft or '(none)'}"
        )
        return AgentOutput(
            intent=intent, confidence=conf, action=action, reasons=reasons, reply=reply, llm_draft=draft,
            guardrail_flags=flags, internal_note=note, lang=lang, pii=pii,
            neighbor_ids=[n.cust_tweet_id for n in neighbors], neighbor_scores=[round(n.score, 3) for n in neighbors],
            neighbor_dm_rate=dm_rate, escalate_prob=p_esc,
            exemplars=[{"cust_text": n.cust_text, "brand_reply": n.brand_reply, "score": round(n.score, 3)} for n in exemplars],
        )
