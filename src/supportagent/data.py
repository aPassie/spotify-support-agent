"""Build the per-brand conversation subset from the raw TWCS dump.

Source data: *Customer Support on Twitter* by Stuart Axelbrooke, CC BY-NC-SA 4.0. Anything
written out of here inherits those terms; see DATA_LICENSE.md.

Unit of analysis: a *root* customer tweet (starts a thread) paired with the
brand's first public reply to it. Follow-up turns are kept for context but the
agent is evaluated on root messages only (see REPORT.md, "what I chose not to build").
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from .textnorm import clean_for_matching, detect_language, display_text, pii_flags

RAW_CSV = Path("data/raw/twcs.csv")
RAW_PARQUET = Path("data/raw/twcs.parquet")
PROCESSED = Path("data/processed")

DM_RE = re.compile(
    r"\b(dm|dms|direct message|private message|pm)\b|dm us|message us|"
    r"send us a (direct|private) message|dm'd|dmed|via dm",
    re.I,
)
URL_RE = re.compile(r"https?://\S+")
SIGNOFF_RE = re.compile(r"\s/[A-Z]{1,3}\b")  # Spotify agent initials e.g. " /JN"


def load_raw(force: bool = False) -> pd.DataFrame:
    """Load the full TWCS dump, caching as parquet after the first read."""
    if RAW_PARQUET.exists() and not force:
        return pd.read_parquet(RAW_PARQUET)
    df = pd.read_csv(
        RAW_CSV,
        dtype={
            "tweet_id": "int64",
            "author_id": str,
            "inbound": str,
            "text": str,
            "response_tweet_id": str,
            "in_response_to_tweet_id": "float64",
        },
    )
    df["inbound"] = df["inbound"].str.strip().str.lower().eq("true")
    df["created_at"] = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    df.to_parquet(RAW_PARQUET)
    return df


def build_brand_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Return one row per customer tweet that received a public reply from `brand`.

    Columns: cust_tweet_id, cust_text, cust_created_at, cust_author,
             brand_reply (earliest), brand_reply_all (all replies to that tweet, joined),
             brand_created_at, is_root, prev_brand_text, prev_cust_text (thread context),
             reply_has_dm, reply_has_url, n_brand_replies
    """
    tid = df.set_index("tweet_id")
    b = df[(~df.inbound) & (df.author_id == brand) & df.in_response_to_tweet_id.notna()].copy()
    b["parent_id"] = b.in_response_to_tweet_id.astype("int64")
    parents = tid.reindex(b.parent_id.values)
    keep = (parents.inbound.eq(True) & parents.text.notna()).values
    b = b[keep]
    parents = parents[keep]

    b = b.sort_values("created_at")
    grp = b.groupby("parent_id", sort=False)
    first = grp.first()
    joined = grp["text"].apply(lambda s: " ||| ".join(s))
    counts = grp.size()

    p = tid.loc[first.index]
    out = pd.DataFrame(
        {
            "cust_tweet_id": first.index.values,
            "cust_text": p.text.values,
            "cust_created_at": p.created_at.values,
            "cust_author": p.author_id.values,
            "brand_reply": first.text.values,
            "brand_reply_all": joined.loc[first.index].values,
            "brand_created_at": first.created_at.values,
            "n_brand_replies": counts.loc[first.index].values,
            "is_root": p.in_response_to_tweet_id.isna().values,
        }
    )
    # thread context for non-root customer tweets: the brand tweet they replied to and
    # the customer tweet before that (two hops up). Plain dict lookups: tweet_id is unique.
    text_of = tid.text.to_dict()
    parent_of = tid.in_response_to_tweet_id.to_dict()

    def _ctx(x):
        if pd.isna(x):
            return None, None
        bid = int(x)
        cid = parent_of.get(bid)
        prev_c = text_of.get(int(cid)) if cid is not None and pd.notna(cid) else None
        return text_of.get(bid), prev_c

    ctx = [_ctx(x) for x in p.in_response_to_tweet_id.values]
    out["prev_brand_text"] = [c[0] for c in ctx]
    out["prev_cust_text"] = [c[1] for c in ctx]

    out["reply_has_dm"] = out.brand_reply_all.str.contains(DM_RE, regex=True).astype(bool)
    out["reply_has_url"] = out.brand_reply_all.str.contains(URL_RE)
    out["cust_has_url"] = out.cust_text.str.contains(URL_RE)
    out = out.sort_values("cust_created_at").reset_index(drop=True)
    return out


def add_message_features(roots: pd.DataFrame, eval_start: str) -> pd.DataFrame:
    """Add language, PII, cleaned text, media-only flag and the temporal split label."""
    r = roots.copy()
    r["text_display"] = r.cust_text.map(display_text)
    r["clean"] = r.cust_text.map(clean_for_matching)
    r["lang"] = r.cust_text.map(detect_language)
    r["pii"] = r.cust_text.map(lambda t: ",".join(pii_flags(t)))
    r["is_media_only"] = (r.clean.str.len() < 3) & r.cust_has_url
    ts = pd.to_datetime(r.cust_created_at)
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    r["cust_created_at"] = ts
    r["split"] = (ts >= pd.Timestamp(eval_start, tz="UTC")).map({True: "eval", False: "history"})
    return r


def load_roots(brand: str = "SpotifyCares") -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{brand}_roots.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brand", default="SpotifyCares")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="re-parse the raw CSV even if a parquet cache exists")
    args = ap.parse_args()
    df = load_raw(force=args.force)
    pairs = build_brand_pairs(df, args.brand)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else PROCESSED / f"{args.brand}_pairs.parquet"
    pairs.to_parquet(out)
    cfg = yaml.safe_load(open("configs/agent.yaml"))
    root = add_message_features(pairs[pairs.is_root], cfg["eval_start"])
    root.to_parquet(PROCESSED / f"{args.brand}_roots.parquet")
    print(f"{args.brand}: {len(pairs)} customer tweets with a public brand reply; {len(root)} are thread roots")
    print(root.split.value_counts().to_dict(), "| languages:", root.lang.value_counts().head(5).to_dict())
    print(f"pii flagged: {(root.pii != '').mean():.1%}; media-only: {root.is_media_only.mean():.1%}")
    print(f"date range: {pairs.cust_created_at.min()} .. {pairs.cust_created_at.max()}")
    print(f"root replies asking for DM: {root.reply_has_dm.mean():.1%}; with URL: {root.reply_has_url.mean():.1%}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
