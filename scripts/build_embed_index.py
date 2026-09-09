"""Embed the historical corpus once and cache it (needs the embedding server on :8082)."""
from pathlib import Path

import yaml

from supportagent.data import load_roots
from supportagent.retrieval import EmbeddingRetriever

cfg = yaml.safe_load(open("configs/agent.yaml"))
roots = load_roots(cfg["brand"])
hist = roots[(roots.split == "history") & (~roots.is_media_only)]
r = EmbeddingRetriever().fit(hist[["cust_text", "brand_reply", "reply_has_dm", "reply_has_url", "cust_tweet_id"]])
r.save(Path("outputs/models/embed_index"))
print(f"embedded {len(hist)} historical tweets -> outputs/models/embed_index.npy {r.X.shape}")
