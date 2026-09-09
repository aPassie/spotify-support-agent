# Decision log

Sixteen choices that could reasonably have gone the other way, in the order I made them.
Every number here is reproduced by `make reproduce`.

1. **Brand chosen from a table, not a hunch.** I profiled the 40 highest-volume brands on volume,
   DM rate, link rate and first-turn share. SpotifyCares has 26k first-turn exchanges, a
   distinctive house style (97% of replies end in agent initials, 66% open with "Hey") and a mixed
   37% DM rate. That mix is the point: enough public resolutions to ground replies on, enough DM
   hand-offs to learn an escalation policy from. Amazon and Hulu almost never DM, so there is
   nothing to learn. T-Mobile and Comcast almost always do, so there is nothing to automate.

2. **Unit of work is a root customer tweet plus the brand's first public reply.** Follow-up turns
   are kept as context columns but not modelled, because later turns are mostly "we've replied to
   your DM" and the content moves into a private channel the dataset does not contain. This choice
   caused one of my two unsafe auto-replies (report, failure mode 4), so it is a real trade, not a
   free simplification.

3. **Temporal split, not random.** Everything before 2017-11-20 is history (retrieval corpus,
   silver labels, training); the golden set is the last two weeks. A random split would let the
   retriever find same-day near-duplicates of eval tweets, and outages produce hundreds of
   identical "is Spotify down?" tweets, which would inflate every number.

4. **The escalation policy is the brand's own DM rate per intent, not my opinion of what is safe.**
   Clustering gave the taxonomy. The brand's historical DM rate per cluster gave the policy. It is
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
   harmless. Uncertainty between playback and billing is the dangerous case. The gate now sums
   probability over escalate-only intents.

7. **Policy thresholds were tuned on historical data, never on the golden set.**
   `scripts/tune_thresholds.py` sweeps thresholds over out-of-fold predictions on the 3,000
   silver-labelled historical tweets, trading auto share against leakage. I took the loosest
   threshold holding leakage at or under 10% (0.40) and a top-1 floor of 0.25, where accuracy on
   items below the floor is 0.22 against 0.72 overall. The sweep is committed so the choice is
   auditable rather than asserted.

8. **A cheap classifier trained on LLM silver labels, instead of the LLM in the loop.** The LLM
   labels 3,000 historical tweets once. TF-IDF plus logistic regression learns from those, then
   runs in microseconds, deterministically, with a probability the policy can threshold. The
   ablation shows what this buys: swapping it for keyword rules keeps precision but collapses
   coverage from 41.9% to 26.3%.

9. **The LLM drafts only auto-handled replies. Escalations get a templated hand-off.** When a human
   will take over, the public reply is one of Spotify's own standard messages, and the LLM draft is
   attached as an internal suggestion. Generating novel public text on cases we have already
   decided we cannot handle is risk with no upside.

10. **Hard guardrails after generation, and no fake agent initials.** Links absent from the
    retrieved exemplars are stripped, handles and initials removed, over-length replies cut at a
    sentence boundary. Any promise of a refund or fix, or request for a password, flips the
    decision to escalate. SpotifyCares signs 97% of replies with a human's initials. The agent
    never does, because imitating a named human is the one style feature a brand should not want
    copied. Result: 0 of 200 replies carry a link the brand never used, and 0 greet a customer by
    name, against 39% for the copy-paste baseline.

11. **Golden set is 160 random plus 40 keyword-targeted, and headline numbers use only the random
    stratum.** Uniform sampling gave three ads complaints, so rare intents needed targeted extras.
    But keyword-matched items are easier for the keyword baseline, so the strata are recorded per
    item and the headline uses the unbiased 160.

12. **Labels record what *should* happen, not what Spotify did.** They disagree on 34 of 200, mostly
    how-to billing and plan questions Spotify took to DM that have a public help-centre answer.
    Copying Spotify would have made the nearest-neighbour baseline look artificially strong.

13. **`praise_or_thanks` was split out of `other`.** The historical clustering showed a distinct
    praise cluster with a much lower DM rate than the rest of `other`. Since `other` always
    escalates, every compliment was heading for a human queue for nothing. The residual `other`
    (venting, jokes, artist and press questions, support-process complaints) still escalates.

14. **The judge is reported but not trusted for ranking.** It is a different model family from the
    generator, gemma judging Qwen, to limit self-preference. It still ranks the copy-paste baseline
    above the agent, because it is anchored on the reference reply and rewards mimicry of it. Worse,
    the no-policy ablation earns the best judge mean of any variant while sending 59 unsafe replies
    instead of 2. So quality claims rest on 60 human ratings and deterministic checks, and the
    judge's agreement statistics are published rather than assumed.

15. **Two bugs found late are written up rather than quietly fixed.** Fast-mode language detection
    labelled 8 of 200 short English tweets as Swedish, German, Dutch, Tagalog or Spanish, which
    would have routed 4% of English customers to a non-existent local team. High-accuracy mode plus
    a confidence floor for short texts fixed all eight. Separately, the embedding ablation first
    "gained" coverage only because dense similarities average 0.86 against TF-IDF's 0.40, so the
    `no_precedent` guard stopped firing; the threshold is now percentile-matched. Both were caught
    by the evaluation, not by reading the code, which is the argument for building the harness
    first.

16. **Banking77 was available and not used.** Its 77 intents are banking-specific (card arrival,
    exchange rates, top-up failures) and none transfers to music streaming, so it would have
    imported a taxonomy that does not fit the traffic. The bigger reason: deriving intents from
    Spotify's own tweets is what lets the escalation policy sit on Spotify's own DM rate per
    intent (decision 4). A borrowed taxonomy breaks that link, which is the argument the whole
    system rests on.

## Setup notes

- Models are local, behind an OpenAI-compatible client, so any hosted endpoint is a two-env-var
  swap. Every call is cached in sqlite keyed by model, messages and sampling parameters, so re-runs
  are free and deterministic at temperature 0.
- Qwen3-4B-Instruct-2507 (generator) and gemma-3-4b-it (judge) on llama.cpp Vulkan, RTX 4060 8 GB,
  one at a time. `--no-mmap` cut host memory from 7.4 GB to 0.5 GB with all layers offloaded,
  which is what made a 14 GB laptop viable.
- Committed: hand labels, human ratings, system and judge outputs, metrics, the LLM cache, the
  2.4 MB classifier, the link allowlist. Not committed: raw dataset, derived parquet, model
  weights, both retrieval indexes.
