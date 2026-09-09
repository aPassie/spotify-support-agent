"""Two baselines the agent must beat.

trivial : intent = majority class; escalation = always escalate; reply = the single most
          common SpotifyCares template (ask for a DM).
simple  : intent = keyword rules; escalation = whether the nearest historical case went to
          DM; reply = the nearest historical SpotifyCares reply, verbatim (handles and agent
          initials stripped).
"""
from __future__ import annotations

import re

from .retrieval import Neighbor
from .textnorm import strip_leading_mentions

DM_LINK = "https://t.co/ldFdZRiNAt"  # Spotify's "message us" deep link, used 7k+ times in the data
TRIVIAL_REPLY = f"Hey! Sorry to hear that. Can you DM us your account's email address or username? We'll take a look backstage {DM_LINK}"

INITIALS_RE = re.compile(r"\s*/[A-Z]{1,3}\b\s*$|\s*/[A-Z]{1,3}\b(?=\s+https?://)")
MULTIPART_RE = re.compile(r"^\s*\d\s*[:/]\s*|\s*\(?\d/\d\)?\s*$")


def clean_brand_reply(text: str) -> str:
    """Historical reply as a reusable template: no @handle, no '/JN' initials, no '1:' part markers."""
    t = strip_leading_mentions(text)
    t = INITIALS_RE.sub(" ", t)
    t = MULTIPART_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def trivial_reply(_text: str) -> str:
    return TRIVIAL_REPLY


def simple_reply(neighbors: list[Neighbor]) -> str:
    if not neighbors:
        return TRIVIAL_REPLY
    return clean_brand_reply(neighbors[0].brand_reply)


def simple_escalate(neighbors: list[Neighbor]) -> bool:
    return bool(neighbors[0].reply_has_dm) if neighbors else True
