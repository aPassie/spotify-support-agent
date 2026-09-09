# Data provenance and licensing

The code in this repository is MIT licensed (see `LICENSE`). The **data files are not**:
they derive from a third-party dataset with a non-commercial, share-alike licence. Read this
before reusing anything under `golden/`, `data/processed/` or `outputs/predictions/`.

## Source

| | |
|---|---|
| Dataset | *Customer Support on Twitter* |
| Author | Stuart Axelbrooke |
| Location | Kaggle `thoughtvector/customer-support-on-twitter` |
| Licence | **CC BY-NC-SA 4.0** (https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| Read from | Hugging Face mirror `SunidhiSriram/twcs` (identical schema; the mirror declares no licence of its own, so the Kaggle terms govern) |

The dataset is already pseudonymised by its author: customer handles are replaced with numeric
ids and email addresses with an `__email__` token. Brand accounts remain named, which is the
point of the exercise. I did not attempt to re-identify anyone and the pipeline never joins
against external data.

## What this repository redistributes

Only a subsample, as the assignment brief invites. The 517 MB source file is **not** committed;
`make data` downloads it.

| file | content | tweets | why it is committed |
|---|---|---|---|
| `golden/golden.jsonl` | 200 customer tweets + the brand's actual reply, with my hand labels | 200 | it *is* the deliverable; the labels are unverifiable without the text |
| `golden/golden_candidates.jsonl` | the same 200 before labelling | 200 | shows the sample was drawn before labels existed |
| `golden/pairwise_agent_vs_embed.csv` | 30 tweets + two candidate replies | 30 | the blinded A/B evidence |
| `data/processed/silver_labels.jsonl` | 3,000 historical tweets + LLM intent labels | 3,000 | the classifier's training data, so the teacher's labels can be audited |
| `outputs/predictions/*.jsonl` | per-item agent output including the retrieved exemplars | ~3,200 (with repeats) | lets a reviewer see what each reply was grounded in without downloading the dataset |

Roughly 6,600 tweet texts in total out of ~2.8 million, about 0.24% of the dataset.

## Terms these files are under

Because CC BY-NC-SA 4.0 is a share-alike licence, derivative data carries the same terms:

- **BY** — attribution to Stuart Axelbrooke, as given above.
- **NC** — **non-commercial use only.** These data files may not be used for commercial
  purposes. This restriction travels with the data, not with the code: the MIT-licensed code can
  be used commercially, but not with this data. Anyone wanting a commercial pipeline should point
  it at their own conversation data.
- **SA** — redistribution of these files, or adaptations of them, must be under
  CC BY-NC-SA 4.0.

## If you would rather not redistribute tweet text at all

Every committed data file carries a `cust_tweet_id`, so it can be reduced to identifiers and
rehydrated by joining against the source. The costs are that `make reproduce` would then require
the 517 MB download, the hand-labelled golden set would no longer be independently reviewable,
and the retrieved exemplars would no longer be inspectable. I judged the 0.24% subsample, under
the original licence and with attribution, to be the better trade for a repository whose purpose
is to let someone check the evaluation.
