import yaml

from supportagent.agent import apply_guardrails
from supportagent.intents import load_taxonomy
from supportagent.policy import decide
from supportagent.retrieval import Neighbor

CFG = yaml.safe_load(open("configs/agent.yaml"))
TAX = load_taxonomy()


def nb(score=0.6, dm=False):
    return Neighbor(0, score, "c", "b", dm, True, 1)


def test_auto_when_public_intent_confident_and_precedent_public():
    d = decide(text="my app keeps crashing", lang="en", pii=[], is_media_only=False, intent="playback_technical", confidence=0.9,
               neighbors=[nb(), nb(), nb(dm=True)], tax=TAX, cfg=CFG)
    assert d.action == "auto" and d.reasons == []


def test_account_intent_escalates_with_reason():
    d = decide(text="charged twice", lang="en", pii=[], is_media_only=False, intent="billing_payment", confidence=0.9,
               neighbors=[nb()], tax=TAX, cfg=CFG)
    assert d.escalate and "needs_account_access" in d.reasons and "payment_dispute" in d.reasons


def test_pii_and_language_escalate():
    d = decide(text="mail bob@x.com", lang="en", pii=["email"], is_media_only=False, intent="playback_technical", confidence=0.9, neighbors=[nb()], tax=TAX, cfg=CFG)
    assert d.primary_reason == "pii_in_public"
    d = decide(text="hola", lang="es", pii=[], is_media_only=False, intent="playback_technical", confidence=0.9, neighbors=[nb()], tax=TAX, cfg=CFG)
    assert d.primary_reason == "language_routing"


def test_low_confidence_and_dm_pattern():
    # top-1 below the floor (0.25) means no usable signal at all
    d = decide(text="x", lang="en", pii=[], is_media_only=False, intent="playback_technical", confidence=0.15,
               neighbors=[nb(dm=True), nb(dm=True), nb()], tax=TAX, cfg=CFG)
    assert "low_confidence" in d.reasons and "historical_dm_pattern" in d.reasons


def test_uncertainty_between_two_public_intents_does_not_escalate():
    """The gate is on escalate-only probability mass, so a low top-1 split across two
    publicly-answerable intents must stay auto-handled."""
    probs = {"content_availability": 0.36, "feature_request_feedback": 0.34, "playback_technical": 0.20,
             "account_access": 0.04, "billing_payment": 0.03, "plan_and_offers": 0.02, "other": 0.01}
    d = decide(text="when is this album coming to spotify", lang="en", pii=[], is_media_only=False,
               intent="content_availability", confidence=0.36, neighbors=[nb(), nb()], tax=TAX, cfg=CFG,
               intent_probs=probs)
    assert d.action == "auto", d.reasons


def test_uncertainty_spread_across_account_intents_escalates():
    """Same low top-1, but the mass now sits on intents that need account access."""
    probs = {"playback_technical": 0.36, "billing_payment": 0.30, "account_access": 0.20,
             "content_availability": 0.10, "plan_and_offers": 0.04}
    d = decide(text="charged and cannot get in", lang="en", pii=[], is_media_only=False,
               intent="playback_technical", confidence=0.36, neighbors=[nb(), nb()], tax=TAX, cfg=CFG,
               intent_probs=probs)
    assert d.escalate and "needs_account_access" in d.reasons


def test_guardrails_remove_unknown_url_and_initials_and_flag_promise():
    reply, flags = apply_guardrails("Hey! Try https://t.co/FAKE111 and we'll refund you /JN", {"https://t.co/real"}, 280)
    assert "https://t.co/FAKE111" not in reply
    assert "/JN" not in reply
    assert {"hallucinated_url", "promise", "removed_initials"} <= set(flags)


def test_guardrails_truncate_long_reply():
    long = "Hey there! " + "This is a sentence. " * 30
    reply, flags = apply_guardrails(long, set(), 280)
    assert len(reply) <= 280 and "too_long" in flags
