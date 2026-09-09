# Decision log

Non-obvious decisions, in the order they were made. Each one is something a reviewer might
reasonably have done differently.

1. **Brand = SpotifyCares, chosen from a table, not a hunch.** I profiled the 40 highest-volume
   brands for volume, share of replies asking for a DM, share containing a link, and share of
   replies that are first-turn. Spotify has 27k first-turn exchanges, a distinctive house style
   (97% of replies end in agent initials, 66% open with "Hey"), and a *mixed* DM rate (37%):
   enough public resolutions to ground replies on, enough DM hand-offs to learn an escalation
   policy from. Amazon/Hulu almost never DM (nothing to learn), T-Mobile/Comcast almost always do
   (nothing to automate).
2. **Unit of work = a root customer tweet + the brand's first public reply.** Follow-up turns
   are kept as context columns but not modelled. Multi-turn threads are mostly "we replied to
   your DM" and would need private-message data we do not have.
3. **Temporal split, not random.** Everything before 2017-11-20 is history (retrieval corpus,
   silver labels, classifier training); the golden set is sampled from the last two weeks. A
   random split would let the retriever find near-duplicates of eval tweets from the same day
   (outages produce hundreds of identical "is Spotify down?" tweets) and inflate every number.
4. **Intents come from clustering the data, and the escalation policy comes from the brand's
   own DM rate per cluster.** Account/billing/plan clusters were taken to DM 80-85% of the
   time; playback/content/feature/ads clusters 8-17%. `auto_handle` in `configs/intents.yaml`
   simply encodes that split. This is the single most important design choice: the policy is
   grounded in observed brand behaviour, not my opinion of what is safe.
5. **Two escalation signals from history, not just the intent.** The agent also looks at the
   k=5 most similar past tweets: if most of them were taken to DM, or none is similar enough,
   it escalates even when the intent is auto-handleable. This catches account-specific
   variants of otherwise public issues.
6. **Cheap classifier trained on LLM silver labels, instead of the LLM in the loop.** 3,000
   historical tweets were labelled once by the 4B model (constrained JSON output, cached),
   then TF-IDF + logistic regression was trained on them. At inference the classifier costs
   microseconds, is deterministic, and gives a probability that the policy can threshold. The
   LLM is only used where it adds value: drafting the reply.
7. **Local open models via an OpenAI-compatible server.** No API keys were available, so the
   pipeline runs Qwen3-4B-Instruct (generator) and gemma-3-4b-it (judge) through llama.cpp on
   CPU. The code only talks to an OpenAI-compatible endpoint, so any hosted model is a two
   env-var change. Every call is cached in sqlite so re-runs are free and deterministic.
8. **Judge from a different model family than the generator.** LLM judges prefer their own
   family's outputs; using gemma to judge Qwen drafts (and Spotify's own historical text) limits
   that. Both are small, which is exactly why the judge is validated against human ratings.
9. **The LLM drafts only the auto-handled replies; escalations get a templated hand-off.** When
   a human will take over, the public reply is one of Spotify's own standard messages (ask for a
   DM, "delete that tweet, it has personal info", "we can help in English or via local
   support"). The LLM draft is still produced and attached as an internal suggestion.
10. **Hard guardrails after generation.** Links must appear in the retrieved exemplars (no
    invented t.co links), agent initials and @handles are stripped, replies over 280 chars are
    cut at a sentence boundary, and any promise of a refund/fix or request for a password flips
    the decision to escalate. The judge would catch most of these, but a guardrail is free and
    deterministic.
11. **No fake agent initials.** SpotifyCares signs every tweet with a human's initials ("/JN").
    The agent never signs that way: imitating a human signature is a trust problem, and it is the
    one style rule the brand would not want copied.
12. **Golden set = 160 random + 40 keyword-targeted.** Uniform sampling gave three ads
    complaints. Targeted extras give each class a usable count, but they are easier for the
    keyword baseline, so headline numbers use the random stratum only and the strata are
    recorded per item.
13. **Labels record what *should* happen, not what Spotify did.** In 34/200 cases my
    `should_escalate` label disagrees with whether Spotify asked for a DM (mostly how-to billing
    and plan questions that have a public help-centre answer). Copying Spotify would have made the
    "simple" baseline look artificially good.
14. **Language detection was changed after it failed on the golden set.** The fast mode flagged
    8/200 short English tweets as Swedish/German/Dutch/Tagalog/Spanish, which would have routed 4%
    of English customers away. High-accuracy mode plus a confidence rule for short texts fixed
    all eight; the failure is kept in the report because it was found by the golden set doing its job.
15. **The models run locally on a laptop GPU, not on CPU and not on a paid API.** The first
    version ran Qwen3-4B on 16 CPU threads at 15 tokens/s, which put the silver-labelling pass at
    40 minutes and made iteration painful. Moving to llama.cpp's Vulkan build on an RTX 4060
    (8 GB) took it to about 8 labels/second. Two details mattered: `--no-mmap` cut the server's
    host memory from 7.4 GB to 0.5 GB on a 14 GB machine, and the generator and judge are run one
    at a time so a single 8 GB card is enough.

16. **The escalation gate thresholds probability mass, not top-1 confidence.** The first version
    escalated when top-1 confidence fell below 0.55. An 8-item debug run showed the flaw: a
    content-availability tweet with a perfect draft was escalated at 0.36 confidence purely because
    a 10-class TF-IDF model rarely exceeds 0.55. The quantity that matters is P(the intent is one
    that needs a human), so the gate now sums the probability on escalate-only intents. Uncertainty
    between two publicly-answerable intents no longer costs coverage.
17. **Both policy thresholds were tuned on historical data, never on the golden set.**
    `scripts/tune_thresholds.py` sweeps the threshold over 5-fold out-of-fold predictions on the
    3,000 silver-labelled historical tweets and reports auto share against leakage. I picked the
    loosest threshold holding leakage at or under 10%, which gave 0.40, and a top-1 floor of 0.25
    where accuracy on the items below it is 0.22 against 0.72 overall. The sweep is committed so
    the choice is auditable rather than asserted.
18. **`praise_or_thanks` was split out of `other` so the agent can answer a compliment.**
    The historical clustering showed a distinct praise cluster whose DM rate is far below the rest
    of `other`. With `other` set to always escalate, every compliment was heading for a human queue
    for no reason. This cost a silver relabel and moved 4 golden items; the residual `other`
    (venting, jokes, artist and press questions, support-process complaints) still escalates.
19. **Language detection uses high-accuracy mode plus a confidence floor for short texts.** The
    fast mode labelled 8 of 200 golden tweets as Swedish, German, Dutch, Tagalog or Spanish when
    they were short English tweets, which would have routed 4% of English customers to a
    non-existent local team. This was caught by the golden set, so the golden set is not fully
    held out with respect to that one component, and the report says so.
20. **The LLM judge is reported but not trusted for ranking, and a deterministic check replaced
    it for the claim that matters.** The judge ranks the copy-paste baseline above the agent while
    I rank it last; it is anchored on the reference reply and rewards stylistic mimicry. A
    zero-judgement regex found what it missed: the copy-paste baseline greets 39% of customers by a
    name copied from a different customer, and the agent does so 0% of the time. Deterministic
    checks now carry the safety claim; the judge is kept as a coarse triage signal with its
    agreement statistics published.
21. **Cached model outputs, the hand labels and the human ratings are committed; the weights,
    parquet files and the 44 MB retrieval index are not.** `make reproduce` recomputes every
    published number in under a minute from committed artefacts, and `make data train` rebuilds
    the rest deterministically. `train.py` also exports the 880 links SpotifyCares actually used,
    so the reply guardrail check runs without any large file.
