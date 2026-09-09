"""Intent taxonomy, baselines and the intent classifier.

Three ways to get an intent label, compared in the report:
  * RulesClassifier   - keyword rules from configs/intents.yaml (the "simple" baseline)
  * llm_label         - zero-shot LLM labelling with constrained JSON output (also used to
                        produce *silver* training labels on historical messages)
  * TfidfClassifier   - TF-IDF (word + char n-grams) + logistic regression trained on
                        silver labels; cheap, deterministic, gives calibrated-ish confidence
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .llm import LLMClient, parse_json
from .textnorm import clean_for_matching, display_text


@dataclass
class Intent:
    name: str
    description: str
    auto_handle: bool
    why: str
    seed_keywords: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass
class Taxonomy:
    brand: str
    intents: list[Intent]
    escalation_reasons: dict[str, str]

    @property
    def names(self) -> list[str]:
        return [i.name for i in self.intents]

    def get(self, name: str) -> Intent:
        return next(i for i in self.intents if i.name == name)

    @property
    def auto_handle_set(self) -> set[str]:
        return {i.name for i in self.intents if i.auto_handle}


def load_taxonomy(path: str | Path = "configs/intents.yaml") -> Taxonomy:
    raw = yaml.safe_load(open(path))
    intents = [Intent(**{k: v for k, v in d.items()}) for d in raw["intents"]]
    return Taxonomy(raw["brand"], intents, raw["escalation_reasons"])


# --------------------------------------------------------------------------- rules
class RulesClassifier:
    """Keyword-hit voting. Deliberately simple: this is the baseline the model must beat."""

    def __init__(self, tax: Taxonomy):
        self.tax = tax
        self.patterns = {
            i.name: [re.compile(r"(?<![a-z])" + re.escape(str(k).lower()) + r"(?![a-z])") for k in i.seed_keywords]
            for i in tax.intents
        }

    def predict_one(self, text: str) -> tuple[str, float]:
        t = clean_for_matching(text).lower()
        hits = {name: sum(1 for p in pats if p.search(t)) for name, pats in self.patterns.items()}
        total = sum(hits.values())
        if total == 0:
            return "other", 0.0
        best = max(hits, key=lambda n: (hits[n], -self.tax.names.index(n)))
        return best, hits[best] / total

    def predict(self, texts) -> list[tuple[str, float]]:
        return [self.predict_one(t) for t in texts]

    def predict_proba(self, texts):
        """Keyword-hit shares as a pseudo-distribution, so the same policy can consume it.

        Used only by the `rules_clf` ablation: it lets us swap the classifier while holding the
        escalation policy fixed, which is the point of that ablation.
        """
        import numpy as np

        classes = self.tax.names
        P = np.zeros((len(texts), len(classes)), dtype=float)
        for r, t in enumerate(texts):
            low = clean_for_matching(t).lower()
            hits = np.array([sum(1 for p in self.patterns[c] if p.search(low)) for c in classes], dtype=float)
            if hits.sum() == 0:
                P[r, classes.index("other")] = 1.0
            else:
                P[r] = hits / hits.sum()
        return P, classes


# --------------------------------------------------------------------------- LLM labelling
def taxonomy_prompt(tax: Taxonomy) -> str:
    lines = [f"You label customer tweets sent to {tax.brand} (Spotify's support account) with exactly one intent.", "", "Intents:"]
    for i in tax.intents:
        lines.append(f"- {i.name}: {' '.join(i.description.split())}")
        for ex in i.examples[:2]:
            lines.append(f'    e.g. "{ex}"')
    lines += [
        "",
        "Rules: pick the single intent that best matches the customer's main problem or request.",
        "If the tweet is not in English, still label the underlying issue if you can, else 'other'.",
        "If nothing actionable is asked (venting, praise, jokes, spam) use 'other'.",
        "Ads shown to a paying Premium user is billing_payment, not ads_complaint.",
        "A missing or greyed-out song is content_availability; a song that exists but will not play is playback_technical.",
        'Answer with JSON only: {"intent": "<name>"}',
    ]
    return "\n".join(lines)


def label_schema(tax: Taxonomy) -> dict:
    return {
        "type": "object",
        "properties": {"intent": {"type": "string", "enum": tax.names}},
        "required": ["intent"],
        "additionalProperties": False,
    }


def llm_label(client: LLMClient, tax: Taxonomy, texts: list[str], desc: str = "llm-label") -> list[str]:
    sys_prompt = taxonomy_prompt(tax)
    schema = label_schema(tax)
    batches = [
        dict(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"Tweet: {display_text(t)}"}],
            max_tokens=30,
            temperature=0.0,
            json_schema=schema,
            tag="label",
        )
        for t in texts
    ]
    outs = client.chat_many(batches, desc=desc)
    labels = []
    for o in outs:
        try:
            lab = parse_json(o)["intent"]
            labels.append(lab if lab in tax.names else "other")
        except Exception:  # noqa: BLE001
            labels.append("other")
    return labels


# --------------------------------------------------------------------------- TF-IDF + LR
class TfidfClassifier:
    def __init__(self, C: float = 4.0):
        feats = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
            ]
        )
        self.pipe = Pipeline([("feats", feats), ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])

    @staticmethod
    def _prep(texts) -> list[str]:
        return [clean_for_matching(t).lower() for t in texts]

    def fit(self, texts, labels) -> "TfidfClassifier":
        self.pipe.fit(self._prep(texts), labels)
        return self

    def predict_proba(self, texts) -> tuple[np.ndarray, list[str]]:
        P = self.pipe.predict_proba(self._prep(texts))
        return P, list(self.pipe.classes_)

    def predict(self, texts) -> list[tuple[str, float]]:
        P, classes = self.predict_proba(texts)
        idx = P.argmax(axis=1)
        return [(classes[i], float(P[r, i])) for r, i in enumerate(idx)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "TfidfClassifier":
        return joblib.load(path)


def read_jsonl(path: Path | str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path | str, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
