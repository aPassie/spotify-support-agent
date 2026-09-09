"""Text cleaning, PII detection and language identification for tweets."""
from __future__ import annotations

import html
import re
from functools import lru_cache

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
LEADING_MENTIONS_RE = re.compile(r"^(?:@\w+\s*)+")
WS_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b|__email__", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")
LONG_DIGITS_RE = re.compile(r"\b\d{9,}\b")  # card / account / order numbers
DM_RE = re.compile(
    r"\b(dm|dms|direct message|private message|pm)\b|dm us|message us|"
    r"send us a (direct|private) message|dm'd|dmed|via dm",
    re.I,
)


def unescape(text: str) -> str:
    return html.unescape(text or "")


def strip_leading_mentions(text: str) -> str:
    """Remove the @handles that start a reply tweet (they are anonymised ids anyway)."""
    return LEADING_MENTIONS_RE.sub("", unescape(text)).strip()


def clean_for_matching(text: str) -> str:
    """Lower-noise version for retrieval / classification: no urls, no handles, unescaped."""
    t = unescape(text)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()


def display_text(text: str) -> str:
    """What the model sees: leading handles removed, other handles kept, urls replaced by a token."""
    t = strip_leading_mentions(text)
    t = URL_RE.sub("[link]", t)
    return WS_RE.sub(" ", t).strip()


def pii_flags(text: str) -> list[str]:
    """Return a list of PII types found in a customer message."""
    t = unescape(text)
    flags = []
    if EMAIL_RE.search(t):
        flags.append("email")
    m = PHONE_RE.search(URL_RE.sub(" ", t))
    if m and sum(ch.isdigit() for ch in m.group(0)) >= 9 and len(m.group(0).split()) <= 4:
        flags.append("phone")
    if LONG_DIGITS_RE.search(URL_RE.sub(" ", t)):
        flags.append("long_number")
    return flags


def extract_urls(text: str) -> list[str]:
    return URL_RE.findall(unescape(text or ""))


_LANGS = [
    "ENGLISH", "SPANISH", "PORTUGUESE", "FRENCH", "GERMAN", "ITALIAN", "INDONESIAN",
    "DUTCH", "TURKISH", "SWEDISH", "TAGALOG", "JAPANESE", "ARABIC", "POLISH", "MALAY",
    "THAI", "RUSSIAN", "KOREAN", "CHINESE", "HINDI", "VIETNAMESE", "GREEK",
]
NON_LATIN_RE = re.compile(r"[\u0370-\u03FF\u0400-\u04FF\u0590-\u06FF\u0900-\u097F\u0E00-\u0E7F\u1100-\u11FF\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]")


@lru_cache(maxsize=1)
def _detector():
    from lingua import Language, LanguageDetectorBuilder

    langs = [getattr(Language, name) for name in _LANGS]
    # NOT low-accuracy mode: on the golden set it flagged 8/200 short English tweets as
    # Swedish/German/Dutch/...; the default mode is still fast (~1.7k tweets/s) and got them all right.
    return LanguageDetectorBuilder.from_languages(*langs).build()


MIN_NON_EN_CONFIDENCE = 0.5


def detect_language(text: str) -> str:
    """ISO-639-1 code; 'und' when the text is too short to judge. Defaults to 'en' when unsure.

    A non-English verdict is only trusted when the detector's confidence is >= 0.5; short
    English tweets otherwise get routed away from the agent for no reason.
    """
    t = clean_for_matching(text)
    if len(t) < 8:
        return "und"
    letters = [c for c in t if c.isalpha()]
    if letters and sum(bool(NON_LATIN_RE.match(c)) for c in letters) / len(letters) > 0.5:
        vals = _detector().compute_language_confidence_values(t)
        return vals[0].language.iso_code_639_1.name.lower() if vals else "xx"
    vals = _detector().compute_language_confidence_values(t)
    if not vals:
        return "en"
    top = vals[0]
    code = top.language.iso_code_639_1.name.lower()
    if code != "en" and top.value < MIN_NON_EN_CONFIDENCE:
        return "en"
    return code
