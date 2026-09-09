# Ablations in detail

Companion to `REPORT.md` section 5. Reproduce with `make ablate` (needs the generator server, and
the embedding server for the `embed` variant), then `make eval`.

## What each variant changes

| variant | change | everything else |
|---|---|---|
| `no_policy` | auto-handle every tweet | classifier, retrieval, drafting, guardrails unchanged |
| `rules_clf` | keyword classifier in place of the silver-trained one | policy thresholds unchanged |
| `no_exemplars` | draft prompt carries style rules but no retrieved cases | decisions unchanged by construction |
| `embed` | bge-small-en-v1.5 dense retrieval in place of TF-IDF | percentile-matched similarity threshold |

## Full numbers

See `outputs/metrics/tables.md` for the generated tables, including per-variant judge check-level
pass rates and paired McNemar tests against the full agent on the random stratum.

### Did embedding retrieval fix failure mode 1? Probably, and I cannot prove it

Swapping TF-IDF for bge-small-en-v1.5 (dense retrieval over the same 19,551 exchanges) fixes the
retrieval mismatches qualitatively. On g027 (*"I updated iOS yesterday and ever since ... it will
randomly pause it"*), TF-IDF's nearest case was an iOS complaint about *missing songs* at
similarity 0.34, and the draft asked about missing songs; the embedding retriever found *"my app
and phone are up to date but the Spotify keeps pausing randomly"* at 0.92 and the draft asks for
the iOS and Spotify versions, which is the right step. On g081, TF-IDF matched on the word
"Christmas" and produced a generic feedback acknowledgement; embeddings matched *"please bring
back the playlist feature on the Roku app"* and the reply names Roku.

Two caveats stop this from being a win.

**The apparent coverage gain was an artefact I had to remove.** Dense cosine similarities here
average 0.86 against TF-IDF's 0.40, so with the TF-IDF threshold of 0.25 the `no_precedent`
safety guard never fired at all (0 escalations against 7) and the variant "gained" coverage purely
by switching off a check. After percentile-matching the threshold (0.763, excluding the same
bottom 4.5% of similarities) the coverage difference shrinks to 41.9% against 43.8%, which is
inside the noise. Reporting the uncalibrated version would have been a straightforward way to
overstate a result by 9% relative.

**A blinded paired comparison cannot separate them at this sample size.** I rated 30 of the 78
items where the two variants produce different replies, presented as A/B in randomised order with
the system hidden. Embeddings won 13, TF-IDF won 8, 9 ties: a 61.9% win rate among decided pairs,
95% CI [0.384, 0.819], p = 0.38. At that effect size an 80%-powered test needs 137 decided pairs,
and only 78 items differ at all, so **this golden set cannot adjudicate retrieval variants no
matter how many of them I rate.** The judge is no help either: it puts embeddings slightly
*behind* TF-IDF (3.46 against 3.52) with heavily overlapping intervals.

So the honest conclusion is: the mechanism is right, the examples are convincing, the measurement
is underpowered, and shipping this would need a larger evaluation set built for the purpose.


## Why the threshold had to be recalibrated

`configs/agent.yaml` carries two similarity thresholds, `min_similarity` (0.25, TF-IDF) and
`min_similarity_embed` (0.763). They are not arbitrary: 0.25 excludes the bottom 4.5% of TF-IDF
top-1 similarities on the golden set, and 0.763 is the value that excludes the same bottom 4.5% of
embedding top-1 similarities. Without that, the `no_precedent` guard silently stops firing when
you change retriever, and the variant looks better because it is doing less checking. This is the
kind of comparison bug that survives peer review because both numbers are individually correct.

## Reproducing the pairwise comparison

```bash
.venv/bin/python -m supportagent.pairwise make-sheet --a agent --b embed --n 30
# fill in the `better` column with A, B or T; the system behind each side is in the .key.json
.venv/bin/python -m supportagent.pairwise score
```

The sheet is randomised per item so A is not consistently one system, and the key is written to a
separate file. My filled sheet is committed as `golden/pairwise_agent_vs_embed.csv`.
