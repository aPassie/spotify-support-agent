# SpotifyCares support agent (Customer Support on Twitter)

An AI support agent for one brand from the *Customer Support on Twitter* dataset. For each
incoming customer tweet it does three things:

1. **Classifies** it into one of 11 intents defined from the data (`configs/intents.yaml`).
2. **Drafts a public reply** grounded in how SpotifyCares historically resolved similar issues,
   retrieved from 19,551 past exchanges.
3. **Decides** auto-handle or escalate to a human, and states the reason.

The brand is **SpotifyCares**, picked from a profile of the 40 highest-volume brands
(see `DECISIONS.md` #1). The interesting claim is not the system, it is the evaluation, so
start with `REPORT.md`.

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
make test        # 10 unit tests: text normalisation, escalation policy, guardrails
```

## Regenerate everything from raw data (~35 min on a laptop GPU)

Needs the 517 MB dataset and a local LLM server. Two 4B models run on an 8 GB GPU one at a time.

```bash
make data        # download twcs.csv, build the SpotifyCares subset (~3 min)
make llm-gen     # download llama.cpp + Qwen3-4B-Instruct-2507 Q4_K_M, serve on :8080
make silver      # 3,000 LLM intent labels on historical tweets (~6 min on RTX 4060)
make train       # fit retriever + intent classifier (~1 min)
make run         # agent + both baselines over the 200 golden tweets (~1.5 min)
make llm-stop && make llm-judge   # swap in gemma-3-4b-it on :8081 (different model family)
make judge       # LLM-as-judge over all three systems (~10 min)
make eval
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
  evaluate.py              all metrics, tables, figure, failure dump
golden/golden.jsonl        200 hand-labelled tweets (intent, escalate, reason, notes)
golden/LABELING.md         sampling and labelling protocol, tie-break rules
golden/human_ratings.csv   60 human reply ratings used to validate the judge
scripts/setup_llm.sh       fetch llama.cpp + models, start a server
scripts/tune_thresholds.py policy threshold selection on historical data only
outputs/metrics/tables.md  every number quoted in the report
REPORT.md                  problem framing, results, failure analysis, caveats
DECISIONS.md               15 non-obvious decisions and why
```

## What is committed and what is not

Committed: hand labels, human ratings, all system outputs, judge outputs, metrics, the LLM cache,
the intent classifier (2.4 MB) and the historical link allowlist. Not committed: the raw dataset,
the derived parquet files, the model weights and the 44 MB retrieval index; `make data` and
`make train` rebuild them deterministically.

## Models used

| role | model | why |
|---|---|---|
| reply drafting, silver labels | Qwen3-4B-Instruct-2507 Q4_K_M | strong instruction following at 4B; runs in 3.6 GB VRAM |
| LLM judge | gemma-3-4b-it Q4_K_M | different model family from the generator, to limit self-preference |

Both served by llama.cpp (Vulkan) on an RTX 4060 Laptop, 8 GB. No paid API was used.

## Credits

- Dataset: *Customer Support on Twitter* by Stuart Axelbrooke (Kaggle `thoughtvector/customer-support-on-twitter`),
  read from the Hugging Face mirror `SunidhiSriram/twcs`.
- Inference: [llama.cpp](https://github.com/ggml-org/llama.cpp). Models: Qwen3-4B-Instruct-2507
  (Alibaba, GGUF by Unsloth), gemma-3-4b-it (Google, GGUF by Unsloth).
- Libraries: scikit-learn (TF-IDF, logistic regression, metrics), lingua (language id),
  rouge-score, pandas, matplotlib, openai (client only).
- The escalation-reason taxonomy, the labelling protocol and all prompts are my own.
