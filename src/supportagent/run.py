"""Run the agent and both baselines over the golden set; write outputs/predictions/*.jsonl."""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import yaml

from .agent import Agent
from .baselines import TRIVIAL_REPLY, simple_escalate, simple_reply
from .intents import RulesClassifier, TfidfClassifier, load_taxonomy, read_jsonl, write_jsonl
from .llm import LLMClient
from .retrieval import Retriever

PRED = Path("outputs/predictions")
MODELS = Path("outputs/models")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="golden/golden.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offline", action="store_true", help="use only cached LLM outputs")
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/agent.yaml"))
    tax = load_taxonomy()
    gold = read_jsonl(Path(args.golden))[: args.limit]
    retr = Retriever.load(MODELS / "retriever.joblib")
    clf = TfidfClassifier.load(MODELS / "intent_clf.joblib")
    rules = RulesClassifier(tax)
    client = LLMClient(parallel=args.parallel, offline=args.offline)
    agent = Agent(tax, cfg, retr, clf, client)

    # LLM drafts are independent, so warm the cache concurrently, then run the (cheap) rest.
    def _one(g):
        return agent.handle(g["cust_text"], lang=g["lang"], tweet_id=g["cust_tweet_id"])

    import concurrent.futures as cf
    from tqdm import tqdm

    with cf.ThreadPoolExecutor(args.parallel) as ex:
        outs = list(tqdm(ex.map(_one, gold), total=len(gold), desc="agent", mininterval=5))

    majority = collections.Counter(r["intent"] for r in read_jsonl(Path("data/processed/silver_labels.jsonl"))).most_common(1)[0][0]
    agent_rows, trivial_rows, simple_rows = [], [], []
    for g, o in zip(gold, outs):
        base = {"id": g["id"], "cust_tweet_id": g["cust_tweet_id"]}
        agent_rows.append(base | o.to_dict())
        trivial_rows.append(base | {"intent": majority, "confidence": 1.0, "action": "escalate", "reasons": ["always"], "reply": TRIVIAL_REPLY})
        nb = retr.query(g["cust_text"], k=cfg["retrieval"]["k"], exclude_tweet_id=g["cust_tweet_id"])
        r_intent, r_conf = rules.predict_one(g["cust_text"])
        simple_rows.append(
            base | {"intent": r_intent, "confidence": r_conf, "action": "escalate" if simple_escalate(nb) else "auto", "reasons": ["nearest_neighbor_dm"] if simple_escalate(nb) else [], "reply": simple_reply(nb), "neighbor_ids": [n.cust_tweet_id for n in nb]}
        )
    write_jsonl(PRED / "agent.jsonl", agent_rows)
    write_jsonl(PRED / "baseline_trivial.jsonl", trivial_rows)
    write_jsonl(PRED / "baseline_simple.jsonl", simple_rows)
    print(f"agent actions: {collections.Counter(r['action'] for r in agent_rows)}")
    print(f"agent reasons: {collections.Counter(r['reasons'][0] for r in agent_rows if r['reasons']).most_common()}")
    print(f"guardrail flags: {collections.Counter(f for r in agent_rows for f in r['guardrail_flags']).most_common()}")
    print(f"llm calls={client.n_calls} cache hits={client.n_cache_hits}")


if __name__ == "__main__":
    main()
