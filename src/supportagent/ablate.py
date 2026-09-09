"""Ablations: hold everything fixed and remove one component at a time.

Answers "which part of this system earns its place?" for the four design choices that cost
the most complexity: retrieval grounding, retrieval quality, the silver-trained classifier,
and the escalation policy.

  no_exemplars  LLM drafts with style instructions only, no retrieved cases
  embed         dense (bge-small) retrieval instead of TF-IDF
  rules_clf     keyword classifier instead of the silver-trained one, policy unchanged
  no_policy     auto-handle everything (drafting and retrieval unchanged)
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
from pathlib import Path

import yaml
from tqdm import tqdm

from .agent import Agent
from .intents import RulesClassifier, TfidfClassifier, load_taxonomy, read_jsonl, write_jsonl
from .llm import LLMClient
from .retrieval import EmbeddingRetriever, Retriever

MODELS = Path("outputs/models")
PRED = Path("outputs/predictions")
VARIANTS = ("no_exemplars", "embed", "rules_clf", "no_policy")


def build(variant: str, tax, cfg, client):
    """Return (agent, cfg) for a variant; cfg may be adjusted so the comparison stays fair."""
    retr = Retriever.load(MODELS / "retriever.joblib")
    clf = TfidfClassifier.load(MODELS / "intent_clf.joblib")
    kw = {}
    if variant == "no_exemplars":
        kw["n_exemplars"] = 0
    elif variant == "embed":
        retr = EmbeddingRetriever.load(MODELS / "embed_index")
        # use the percentile-matched threshold, see configs/agent.yaml
        cfg = {**cfg, "retrieval": {**cfg["retrieval"], "min_similarity": cfg["retrieval"]["min_similarity_embed"]}}
    elif variant == "rules_clf":
        clf = RulesClassifier(tax)
    elif variant == "no_policy":
        kw["force_auto"] = True
    else:
        raise SystemExit(f"unknown variant {variant}")
    return Agent(tax, cfg, retr, clf, client, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--golden", default="golden/golden.jsonl")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/agent.yaml"))
    tax = load_taxonomy()
    gold = read_jsonl(Path(args.golden))
    client = LLMClient(parallel=args.parallel, offline=args.offline)
    for variant in args.variants:
        agent = build(variant, tax, cfg, client)
        with cf.ThreadPoolExecutor(args.parallel) as ex:
            outs = list(
                tqdm(ex.map(lambda g: agent.handle(g["cust_text"], lang=g["lang"], tweet_id=g["cust_tweet_id"]), gold), total=len(gold), desc=variant, mininterval=5)
            )
        rows = [{"id": g["id"], "cust_tweet_id": g["cust_tweet_id"]} | o.to_dict() for g, o in zip(gold, outs)]
        write_jsonl(PRED / f"ablation_{variant}.jsonl", rows)
        print(f"{variant}: {collections.Counter(r['action'] for r in rows)}  -> ablation_{variant}.jsonl")
    print(f"llm calls={client.n_calls} cache hits={client.n_cache_hits}")


if __name__ == "__main__":
    main()
