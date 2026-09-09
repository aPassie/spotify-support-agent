# Decision log

Sixteen non-obvious decisions, in the order they were made. Each is something a reviewer might
reasonably have decided differently. Numbers referenced here are reproduced by `make reproduce`.

1. **Brand chosen from a table, not a hunch.** I profiled the 40 highest-volume brands on volume,
   DM rate, link rate and first-turn share. SpotifyCares has 26k first-turn exchanges, a
   distinctive house style (97% of replies end in agent initials, 66% open with "Hey") and a
   *mixed* 37% DM rate: enough public resolutions to ground replies on, enough DM hand-offs to
   learn an escalation policy from. Amazon and Hulu almost never DM (nothing to learn),
   T-Mobile and Comcast almost always do (nothing to automate).

2. **Unit of work is a root customer tweet plus the brand's first public reply.** Follow-up turns
   are kept as context columns but not modelled: later turns are mostly "we've replied to your DM",
   so the content moves into a private channel the dataset does not contain. This decision
   directly caused one of my two unsafe auto-replies (report, failure mode 4), so it is a real
   trade rather than a free simplification.

3. **Temporal split, not random.** Everything before 2017-11-20 is history (retrieval corpus,
   silver labels, training); the golden set is the last two weeks. A random split would let the
   retriever find same-day near-duplicates of eval tweets, and outages produce hundreds of
   identical "is Spotify down?" tweets, which would inflate every number.

4. **The escalation policy is the brand's own DM rate per intent, not my opinion of what is safe.**
   Clustering gave the taxonomy; the brand's historical DM rate per cluster gave the policy. It is
   bimodal with a clean gap: billing 0.92, account 0.82, plans 0.78 against downloads 0.39, devices
   0.22, playback 0.16, content 0.09, features 0.05. `auto_handle` in `configs/intents.yaml`
   encodes that gap. This is the single design choice the whole system rests on.

5. **Escalation also reads evidence from history, not just the predicted intent.** The agent looks
   at the 5 nearest past cases: if most went to DM, or none is similar enough, it escalates even
   when the intent is auto-handleable. This catches account-specific variants of otherwise public
   issues, and it fires on 19 of 200 items.

6. **The escalation gate thresholds probability mass, not top-1 confidence.** The first version
   escalated below 0.55 top-1 confidence. An 8-item debug run showed the flaw: a
   content-availability tweet with a perfect draft was escalated at 0.36 purely because a 10-class
   TF-IDF model rarely exceeds 0.55. Uncertainty between two publicly-answerable intents is
   harmless; uncertainty between playback and billing is the dangerous case. The gate now sums
   probability over escalate-only intents.

7. **Policy thresholds were tuned on historical data, never on the golden set.**
   `scripts/tune_thresholds.py` sweeps thresholds over out-of-fold predictions on the 3,000
   silver-labelled historical tweets, trading auto share against leakage. I took the loosest
   threshold holding leakage at or under 10% (0.40) and a top-1 floor of 0.25, where accuracy on
   items below the floor is 0.22 against 0.72 overall. The sweep is committed so the choice is
   auditable rather than asserted.

8. **A cheap classifier trained on LLM silver labels, instead of the LLM in the loop.** The LLM
   labels 3,000 historical tweets once; TF-IDF plus logistic regression learns from those and then
   runs in microseconds, deterministically, with a probability the policy can threshold. The
   ablation shows what this buys: swapping it for keyword rules keeps precision but collapses
   coverage from 41.9% to 26.3%.

9. **The LLM drafts only auto-handled replies; escalations get a templated hand-off.** When a human
   will take over, the public reply is one of Spotify's own standard messages, and the LLM draft is
   attached as an internal suggestion. Generating novel public text on cases we have already
   decided we cannot handle is risk with no upside.

10. **Hard guardrails after generation, and no fake agent initials.** Links absent from the
    retrieved exemplars are stripped, handles and initials removed, over-length replies cut at a
    sentence boundary, and any promise of a refund or fix or request for a password flips the
    decision to escalate. SpotifyCares signs 97% of replies with a human's initials; the agent
    never does, because imitating a named human is the one style feature the brand should not want
    copied. Result: 0 of 200 replies carry a link the brand never used, and 0 greet a customer by
    a name, against 39% for the copy-paste baseline.

11. **Golden set is 160 random plus 40 keyword-targeted, and headline numbers use only the random
    stratum.** Uniform sampling gave three ads complaints, so rare intents needed targeted extras;
    but keyword-matched items are easier for the keyword baseline, so the strata are recorded per
    item and the headline is the unbiased 160.

12. **Labels record what *should* happen, not what Spotify did.** They disagree on 34 of 200, mostly
    how-to billing and plan questions Spotify took to DM that have a public help-centre answer.
    Copying Spotify would have made the nearest-neighbour baseline look artificially strong.

13. **`praise_or_thanks` was split out of `other`.** The historical clustering showed a distinct
    praise cluster with a much lower DM rate than the rest of `other`. Since `other` always
    escalates, every compliment was heading for a human queue for nothing. The residual `other`
    (venting, jokes, artist and press questions, support-process complaints) still escalates.

14. **The judge is reported but not trusted for ranking, and a deterministic check carries the
    safety claim.** The judge is a different model family from the generator (gemma judging Qwen)
    to limit self-preference, and it still ranks the copy-paste baseline above the agent because it
    is anchored on the reference reply and rewards stylistic mimicry. Worse, the no-policy ablation
    scores the best judge mean of any variant while sending 59 unsafe replies instead of 2. So
    quality claims rest on 60 human ratings plus regex checks, and the judge's agreement statistics
    are published rather than assumed.

15. **Two bugs found late are kept in the write-up rather than quietly fixed.** Fast-mode language
    detection labelled 8 of 200 short English tweets as Swedish, German, Dutch, Tagalog or Spanish,
    which would have routed 4% of English customers to a non-existent local team; high-accuracy
    mode plus a confidence floor for short texts fixed all eight. And the embedding ablation
    initially "gained" coverage only because dense cosine similarities average 0.86 against
    TF-IDF's 0.40, so the `no_precedent` guard silently stopped firing; the threshold is now
    percentile-matched. Both were caught by the evaluation rather than by reading the code, which
    is the argument for building the harness first.

16. **Banking77 was offered and not used.** The brief allows it as an optional secondary dataset
    for intent work. Its 77 intents are banking-specific (card arrival, exchange rates, top-up
    failures) and none transfers to music streaming, so pre-training on it would have imported a
    taxonomy that does not fit the traffic. More importantly, deriving intents from Spotify's own
    tweets is what let the escalation policy be grounded in Spotify's own DM behaviour per intent
    (decision 4), which is the argument the whole system rests on. A borrowed taxonomy would have
    broken that link.

## Engineering notes that are not really decisions

- Local open models via an OpenAI-compatible client, so any hosted endpoint is a two-env-var swap;
  every call cached in sqlite keyed by model, messages and sampling parameters, so re-runs are free
  and deterministic at temperature 0.
- Qwen3-4B-Instruct-2507 (generator) and gemma-3-4b-it (judge) on llama.cpp Vulkan, RTX 4060 8 GB,
  run one at a time. `--no-mmap` cut the server's host memory from 7.4 GB to 0.5 GB with all layers
  offloaded, which is what made a 14 GB laptop viable.
- Committed: hand labels, human ratings, all system and judge outputs, metrics, the LLM cache, the
  2.4 MB classifier and the 880-link allowlist. Not committed: raw dataset, derived parquet, model
  weights, the 44 MB retrieval index and the embedding index. `make reproduce` recomputes every
  published number from committed artefacts in under two seconds.
