# SpotifyCares support agent (Customer Support on Twitter)

An AI support agent for one brand from the *Customer Support on Twitter* dataset. For each
incoming customer tweet it does three things:

1. **Classifies** it into one of 11 intents defined from the data (`configs/intents.yaml`).
2. **Drafts a public reply** grounded in how SpotifyCares historically resolved similar issues,
   retrieved from 19,551 past exchanges.
3. **Decides** auto-handle or escalate to a human, and states the reason.

The brand is **SpotifyCares**, picked from a profile of the 40 highest-volume brands
(see `DECISIONS.md` #1). The interesting claim is not the system, it is the evaluation, so
start with **[REPORT.md](REPORT.md)**.

### Where each deliverable lives

| asked for | here |
|---|---|
| Repo with a runnable pipeline, headline reproducible in < 15 min | this repo; `make setup && make reproduce` takes ~2 seconds |
| Golden evaluation set, 150-250 hand-labelled, + note on sampling and labelling | [`golden/golden.jsonl`](golden/golden.jsonl) (200 items), protocol in [`golden/LABELING.md`](golden/LABELING.md) |
| Evaluation harness: automated metrics + LLM-judge rubric + judge/human agreement | `src/supportagent/evaluate.py` and `judge.py`; numbers in [`outputs/metrics/tables.md`](outputs/metrics/tables.md); 60 human ratings in [`golden/human_ratings.csv`](golden/human_ratings.csv) |
| Report: framing, what I did not build, 2+ baselines, top-5 failures, misleading-number section, next week | **[REPORT.md](REPORT.md)** (~6 pages), with [`ABLATIONS.md`](ABLATIONS.md) as an appendix |
| Decision log, 10-15 non-obvious decisions | [`DECISIONS.md`](DECISIONS.md) (16) |

The mandatory *"what is misleading about my headline number?"* section is
[REPORT.md section 7](REPORT.md#7-what-is-misleading-about-my-headline-number).

## Headline result

On 160 randomly sampled held-out tweets, hand-labelled:

| system | auto-handle rate | of those, correctly auto-handled | replies sent that needed a human |
|---|---|---|---|
| trivial baseline (always escalate) | 0% | n/a | 0 |
| simple baseline (nearest past case) | 62.5% | 81.0% | 19 |
| **this agent** | **41.9%** | **97.0%** | **2** |

The agent handles two thirds as much traffic as the copy-the-nearest-case baseline while sending
one tenth as many replies that should have gone to a human. Intent accuracy is 0.681 against
0.494 for the keyword baseline. `REPORT.md` explains why the headline is flattering and which
number I would actually defend.

## Reproduce the headline numbers (< 1 minute, no GPU, no API key, no dataset download)

```bash
make setup       # uv venv + install (only step that needs the network)
make reproduce   # recompute every number in outputs/metrics/ from committed model outputs
```

`make reproduce` reads the committed hand labels (`golden/golden.jsonl`), system outputs
(`outputs/predictions/`), judge outputs (`outputs/judge/`) and my human ratings
(`golden/human_ratings.csv`), and rewrites `outputs/metrics/tables.md`, `metrics.json`, the
confusion figure and `agent_failures.jsonl`. No model is called, so the numbers are exactly the
ones in the report.

```bash
make test        # 35 tests: text normalisation, escalation policy, guardrails, and a check
                 # that every number quoted in REPORT.md still matches outputs/metrics/
```

## Ablations: which components earn their place

| variant | auto-handle rate | auto precision | unsafe replies (of 160) | judge mean |
|---|---|---|---|---|
| **full agent** | 0.419 | 0.970 | **2** | 3.52 |
| no escalation policy | 1.000 | 0.631 | **59** | **3.67** |
| keyword classifier | 0.263 | 0.929 | 3 | 3.38 |
| no retrieved exemplars | 0.419 | 0.970 | 2 | 3.24 |
| dense (embedding) retrieval | 0.438 | 0.971 | 2 | 3.46 |

Note the second row: deleting the safety policy sends 59 unsafe public replies instead of 2 **and
gives the best LLM-judge score of any variant**. Optimising the judge metric would make the system
much worse. Details in `ABLATIONS.md`; reproduce with `make ablate` then `make eval`.

## Regenerate everything from raw data (~35 min on a laptop GPU)

Needs the 517 MB dataset and a local LLM server. Two 4B models run on an 8 GB GPU one at a time.

```bash
make data        # download twcs.csv, build the SpotifyCares subset (~3 min)
make llm-gen     # download llama.cpp + Qwen3-4B-Instruct-2507 Q4_K_M, serve on :8080
make silver      # 3,000 LLM intent labels on historical tweets (~6 min on RTX 4060)
make train       # fit retriever + intent classifier (~1 min)
make run         # agent + both baselines over the 200 golden tweets (~1.5 min)
make llm-stop && make llm-judge   # swap in gemma-3-4b-it on :8081 (different model family)
make judge       # LLM-as-judge over every system (~10 min)
make eval
```

Optional, for the ablations:

```bash
make llm-stop && make llm-embed   # bge-small-en-v1.5 on :8082 (67 MB)
make embed-index                  # embed 19.5k historical tweets (~25 s on GPU)
make llm-gen                      # generator back up, alongside the embedding server
make ablate                       # four variants over the golden set
make thresholds                   # re-derive the policy thresholds on historical data
make pairwise                     # blinded A/B sheet for two systems; fill it in, then pairwise-score
```

Any OpenAI-compatible endpoint works instead of the local server:

```bash
export LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...        # generator
export JUDGE_BASE_URL=... JUDGE_MODEL=...                    # judge
```

Every LLM call is cached in `outputs/cache/llm_cache.sqlite`, keyed by model, messages and
sampling parameters, so re-runs are free and deterministic. Temperature is 0 everywhere.

## Repository layout

```
configs/intents.yaml       intent taxonomy + auto/escalate policy + escalation reason ids
configs/agent.yaml         split date, retrieval k, policy thresholds
src/supportagent/
  data.py                  raw TWCS -> per-brand root tweet + first public reply
  textnorm.py              cleaning, PII detection, language id
  retrieval.py             TF-IDF (word + char) nearest-neighbour over past exchanges
  intents.py               taxonomy, keyword baseline, LLM labeller, TF-IDF classifier
  silver.py                LLM silver labels on historical tweets
  train.py                 fit retriever + classifier, export link allowlist
  policy.py                escalation decision with reasons
  agent.py                 classify -> retrieve -> decide -> draft -> guardrails
  baselines.py             trivial and simple baselines
  run.py                   run all three systems over the golden set
  judge.py / judge_run.py  LLM-as-judge rubric + judge/human agreement
  ablate.py                component ablations (no policy / no exemplars / rules / dense retrieval)
  pairwise.py              blinded A/B preference harness with power analysis
  evaluate.py              all metrics + CIs + McNemar tests, tables, figure, failure dump
golden/golden.jsonl        200 hand-labelled tweets (intent, escalate, reason, notes)
golden/LABELING.md         sampling and labelling protocol, tie-break rules
golden/human_ratings.csv   60 human reply ratings used to validate the judge
golden/pairwise_*.csv      blinded A/B preferences (+ hidden key) for the retrieval comparison
scripts/setup_llm.sh       fetch llama.cpp + models, start a server
scripts/tune_thresholds.py policy threshold selection on historical data only
outputs/metrics/tables.md  every number quoted in the report
LICENSE / DATA_LICENSE.md  MIT for the code; CC BY-NC-SA 4.0 for the redistributed tweet text
REPORT.md                  problem framing, results, failure analysis, caveats
ABLATIONS.md               component ablations and the retrieval comparison in detail
DECISIONS.md               15 non-obvious decisions and why
```

## What is committed and what is not

Committed: hand labels, human ratings, blinded A/B preferences, all system and judge outputs,
metrics, the LLM cache, the intent classifier (2.4 MB) and the historical link allowlist. Not
committed: the raw dataset, derived parquet files, model weights, and the retrieval indexes (44 MB
sparse, 30 MB dense) since they contain dataset text and are regenerable; `make data`, `make train`
and `make embed-index` rebuild them deterministically.

## Models used

| role | model | why |
|---|---|---|
| reply drafting, silver labels | Qwen3-4B-Instruct-2507 Q4_K_M | strong instruction following at 4B; runs in 3.6 GB VRAM |
| LLM judge | gemma-3-4b-it Q4_K_M | different model family from the generator, to limit self-preference |
| dense retrieval ablation | bge-small-en-v1.5 f16 | 67 MB, embeds the 19.5k corpus in 25 s |

All served by llama.cpp on an RTX 4060 Laptop (8 GB) using the prebuilt **Vulkan** binaries, which
work on NVIDIA and AMD with only the normal driver. llama.cpp publishes no prebuilt CUDA build for
Linux (Windows only), so CUDA would mean compiling from source; Vulkan was about 7x the CPU build
here, which was enough. `scripts/setup_llm.sh` prefers a GPU build when present and falls back to
CPU. No paid API was used.

Two practical notes if you run this on a memory-constrained laptop. Pass `--no-mmap` (as
`scripts/setup_llm.sh` does): with all layers offloaded, it cut the server's *host* memory from
7.4 GB to 0.5 GB on a 14 GB machine. And run the generator and judge one at a time via
`make llm-stop`; the Vulkan backend's host memory also grows with request volume, reaching about
7 GB after the ~600-call judge pass, so two servers plus a desktop session will swap.

## Licensing (read this before reusing the data)

Code is **MIT** (`LICENSE`). The data files are **not**: they derive from a dataset licensed
**CC BY-NC-SA 4.0**, so they are non-commercial and share-alike. This repository redistributes
about 6,600 tweet texts, roughly 0.24% of the source, as the brief's "a subsample is expected and
encouraged" invites. Full provenance, the per-file inventory and what the restrictions mean:
`DATA_LICENSE.md`.

## Credits

- Dataset: *Customer Support on Twitter* by Stuart Axelbrooke (Kaggle `thoughtvector/customer-support-on-twitter`,
  CC BY-NC-SA 4.0), read from the Hugging Face mirror `SunidhiSriram/twcs`.
- Inference: [llama.cpp](https://github.com/ggml-org/llama.cpp). Models: Qwen3-4B-Instruct-2507
  (Alibaba, GGUF by Unsloth), gemma-3-4b-it (Google, GGUF by Unsloth).
- Libraries: scikit-learn (TF-IDF, logistic regression, metrics), lingua (language id),
  rouge-score, pandas, matplotlib, openai (client only).
- Not used: Banking77 (`PolyAI/banking77`), offered by the brief as an optional secondary source
  for intent work; see `DECISIONS.md` #16 for why.
- The escalation-reason taxonomy, the labelling protocol and all prompts are my own.
