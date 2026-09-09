# SpotifyCares support agent: what I built, what it is worth, and where it breaks

## 1. Problem framing: what "good" means for this brand

I picked **SpotifyCares** by profiling the 40 highest-volume brands on volume, how often the reply
asks the customer to move to DM, how often it carries a link, and how often it is a thread's first
turn. Spotify sits in a useful middle: 26,000 first-turn exchanges and a **37% DM rate**. Amazon
and Hulu almost never move to DM (nothing to learn about escalation), T-Mobile and Comcast almost
always do (nothing to automate). A brand that does both teaches both halves of the job.

Spotify support on Twitter is not a resolution channel. It is a **triage and deflection** channel.
Across a few hundred exchanges, a good public reply is one of five things: a troubleshooting step,
a help-centre link, an explanation of a limit such as licensing or plan rules, an acknowledgement
that routes feedback onward, or a clean hand-off to DM. So "good" here means:

- **Never send a public reply where a human was needed.** A wrong public reply on a billing or
  login problem is worse than silence: it is public, it is quotable, and it burns the customer's
  second attempt.
- **Deflect the genuinely public issues**, which are the majority of the volume.
- **Say why** when escalating, so the human picking it up starts ahead.

That ordering makes this a **precision problem, not a coverage problem**, and is why my headline
metric is "of the tweets we auto-answered, how many should we have".

### What I chose not to build

- **No multi-turn dialogue.** I model a thread's *root* tweet and the brand's first public reply.
  Later turns here are mostly "we've replied to your DM", so the content moves to a private
  channel the dataset does not contain. Modelling it would mean modelling text I cannot see.
- **No fine-tuning.** With 19.5k exchanges available at inference time, retrieval buys the same
  brand-voice grounding as a LoRA, stays inspectable, and updates when the brand changes its links.
- **No LLM in the classification loop.** The LLM labels 3,000 historical tweets once. A TF-IDF
  and logistic-regression classifier learns from those and runs in microseconds, with a
  probability the policy can threshold. The LLM is used only where it earns its cost: drafting.
- **No agent initials.** SpotifyCares signs 97% of replies with a human's initials ("/JN"). The
  agent never does. Imitating a named human is the one stylistic feature the brand should not want
  copied, and it would make the automation undisclosed.
- **No sentiment model.** Anger is not the escalation trigger here, account state is. An angry
  playback complaint is still publicly answerable. A polite billing question is not.

## 2. The system

Pipeline: language id and PII scan, then an 11-class intent classifier (TF-IDF word plus character
n-grams into logistic regression), then retrieval of the 8 nearest past customer tweets with the
brand's actual replies, then the auto/escalate policy. Auto-handled tweets get an LLM draft
conditioned on 4 exemplars, followed by hard guardrails. Escalated tweets get one of Spotify's own
hand-off templates, with the draft attached as an internal suggestion for the human.

**The taxonomy came from the data.** I clustered the 24,000 English root tweets (TF-IDF, LSA to 120
dims, k-means) and read samples per cluster. Eleven intents survived. Clustering also gave the
escalation policy for free: the brand's own DM rate per cluster is bimodal, confirmed by the silver
labels on 3,000 tweets.

| intent | share of silver labels | brand's historical DM rate | policy |
|---|---|---|---|
| billing_payment | 8.1% | 0.92 | escalate |
| account_access | 10.8% | 0.82 | escalate |
| plan_and_offers | 11.8% | 0.78 | escalate |
| downloads_offline | 3.3% | 0.39 | auto |
| other | 11.8% | 0.28 | escalate |
| device_integration | 2.2% | 0.22 | auto |
| playback_technical | 9.7% | 0.16 | auto |
| ads_complaint | 1.9% | 0.14 | auto |
| content_availability | 15.9% | 0.09 | auto |
| praise_or_thanks | 1.6% | 0.08 | auto |
| feature_request_feedback | 22.8% | 0.05 | auto |

There is a clean gap between 0.39 and 0.78. The policy is that gap. This is the single design
decision the whole system rests on, and it is an observation about the brand, not my opinion.

**The escalation gate uses probability mass, not top-1 confidence.** The first version escalated
whenever the classifier's top-1 confidence fell below a threshold. That is the wrong quantity: being
unsure between `content_availability` and `feature_request_feedback` is harmless because both are
publicly answerable, while being unsure between `playback_technical` and `billing_payment` is
exactly the dangerous case. So the gate is the summed probability on escalate-only intents.
Both thresholds were chosen on held-out **historical** data with `scripts/tune_thresholds.py`,
never on the golden set: 0.40 was the loosest threshold keeping leakage (auto-handled items whose
true intent is escalate-only) at or under 10%.

Escalation also fires on evidence independent of the intent: personal data in a public tweet,
legal or safety language, a possible account compromise, a refund demand, a non-English message,
a media-only tweet, no similar precedent, or most similar past cases having gone to DM.

Every escalation carries a stable reason id, so I can measure *why* the system escalates.

**Guardrails, after generation.** Four checks run on every draft:

- links absent from the retrieved exemplars are stripped (0 of 200 replies carried a link the
  brand had never used)
- handles and agent initials are removed
- replies over 280 characters are cut at a sentence boundary
- a promise of a refund or fix, or a request for a password or card details, flips the decision
  to escalate

## 3. The golden set

200 root tweets from **2017-11-20 to 2017-12-03**, strictly after every tweet used for retrieval,
silver labels or training. 160 are a uniform random sample. The other 40 are keyword-targeted extras so
that rare intents have a usable count. Because the targeted items are easier for a keyword
baseline, **all headline numbers are on the 160-item random stratum**.

Each item carries an intent, a should-escalate decision, a primary escalation reason, whether it is
really English, and an ambiguity note. The decision label is *what should happen*, not what Spotify
did: they disagree on 34 of 200, mostly how-to billing and plan questions Spotify took to DM that
have a public help-centre answer. Copying Spotify's behaviour would have made the nearest-neighbour
baseline look artificially strong. Protocol and tie-break rules: `golden/LABELING.md`.

## 4. Results

Two baselines. **Trivial**: always escalate, majority intent, and Spotify's single most common
template (the "DM us your account email" message, used 7,351 times in the data). **Simple**:
keyword intent, escalate if the nearest past case went to DM, and reply with that nearest
historical reply verbatim.

### The decision that matters (160 random held-out tweets)

| system | auto-handle rate [95% CI] | auto precision [95% CI] | replies sent that needed a human | escalate recall |
|---|---|---|---|---|
| trivial | 0.0% | n/a | 0 of 0 | 1.000 |
| simple | 62.5% [55.6, 70.0] | 0.810 [0.722, 0.875] | **19 of 100** | 0.678 |
| **agent** | **41.9% [34.4, 50.0]** | **0.970 [0.898, 0.992]** | **2 of 67** | **0.966** |

The simple baseline auto-answers half again as much traffic and gets 19 of those wrong. The agent
gives up a third of that coverage to cut the errors to 2. Given that a wrong public reply on a
billing problem costs more than a delayed correct one, that is the trade I would ship. The trivial
baseline is safe by construction and worth zero: it deflects nothing.

Because every system sees the same 160 items, the comparison is paired, so McNemar's exact test
applies. On unsafe replies the agent errs alone on 1 item and the simple baseline on 18
(p = 7.6e-05). On intent the agent errs alone on 15 and the baseline on 45 (p = 1.3e-04). Both
advantages are real and not an artefact of sample size. Against the trivial baseline the unsafe
comparison is meaningless by construction (it never auto-answers), which is the point of including
it: it exists to show that the safety number is trivially gameable by refusing to work.

### Intent classification

| system | accuracy (random) | macro-F1 (random) |
|---|---|---|
| trivial (majority class) | 0.144 | 0.023 |
| simple (keyword rules) | 0.494 | 0.472 |
| agent (TF-IDF + LR on silver labels) | **0.681** | **0.559** |

Per-class F1 on all 200 runs from `account_access` 0.92, `billing_payment` 0.77 and
`plan_and_offers` 0.76 down to `device_integration` and `praise_or_thanks` at 0.40 (full table in
`outputs/metrics/tables.md`). The three escalate-critical classes are the three strongest, which
is the right shape for this policy: the weak classes all sit inside the auto-handle group, where a
confusion costs a mediocre reply rather than an unsafe one.

### Reply quality: the LLM judge, and why I do not trust its ranking

The judge (gemma-3-4b-it, a different family from the Qwen generator) scores five binary checks and
a 1-5 overall, given the tweet, the candidate reply, Spotify's actual reply and three similar past
exchanges.

| system | judge mean | judge % >= 4 | my mean (blind, n=20 each) |
|---|---|---|---|
| trivial | 2.81 | 0.33 | 2.30 |
| simple | **3.61** | **0.62** | **2.15** |
| agent | 3.52 | 0.59 | **2.85** |

**The judge puts the copy-paste baseline first. I put it last.** The mechanism: the judge sees the
real historical reply as a reference, and that baseline *is* a real historical reply, so it matches
the reference's style perfectly. What the judge misses is that the reply belongs to a different
customer. A deterministic check finds it:

| system | greets the customer by a name the system cannot possibly know |
|---|---|
| agent | **0%** |
| trivial | 0% |
| simple | **39%** (78 of 200) |

The dataset is anonymised, so no reply can legitimately know a customer's first name: all 78 are
Spotify greeting the wrong person in public. The judge still rated that baseline `safe` at 0.96.
It also missed replies claiming "we've just replied to your DM" when no DM existed, and one
telling a listener to "reach out to your distributor".

Judge-human agreement over 60 blind ratings: Spearman 0.49, quadratic-weighted kappa 0.32, exact
agreement 0.23, within-one 0.60, and the judge is **+1.02 points more generous**. So it is usable
as a coarse filter (it does separate the trivial baseline from the other two) and unusable as a
ranker between systems of similar surface quality. Every reply-quality claim I make rests on the 60
human ratings and the deterministic checks, not the judge mean.

## 5. Which components earn their place (ablations)

Each variant swaps exactly one component, scored on the same 200 items. `make ablate` reproduces
this; per-variant judge detail is in `outputs/metrics/tables.md`.

| variant | auto-handle rate | auto precision | unsafe replies (of 160) | judge mean | what it tells us |
|---|---|---|---|---|---|
| **full agent** | 0.419 | 0.970 | **2** | 3.52 | - |
| no escalation policy | 1.000 | 0.631 | **59** | **3.67** | the policy is the system |
| keyword classifier instead of silver-trained | 0.263 | 0.929 | 3 | 3.38 | the classifier buys coverage, not safety |
| no retrieved exemplars | 0.419 | 0.970 | 2 | 3.24 | grounding is worth 0.28 judge points |
| embedding retrieval instead of TF-IDF | 0.438 | 0.971 | 2 | 3.46 | see below |

**Removing the escalation policy multiplies unsafe replies by 30 (2 to 59) and *raises* the judge's
score to the best of any variant.** A team optimising the judge number would delete the safety
policy and watch the metric improve.

**The silver-label classifier pays for itself in coverage, not safety.** Keyword rules keep auto
precision within noise (0.929 against 0.970, overlapping CIs) but collapse coverage from 41.9% to
26.3%. Its value is confidently placing a tweet in a publicly-answerable class.

**Retrieval grounding is what makes replies specific.** Removing exemplars leaves every decision
identical (retrieval only feeds the draft) and drops the judge score to 3.24, brand-consistency
from 0.61 to 0.48. Without exemplars the model writes plausible support English that does not match
what Spotify does.

**Embedding retrieval fixes the mismatches on inspection but cannot be shown to help.** Dense
retrieval finds the right precedent where TF-IDF matched on shared vocabulary. Two caveats keep it
out of the headline. First, its apparent coverage gain was an artefact of a similarity threshold
not recalibrated for the new scale, which I corrected. Second, a blinded paired comparison of 30
of the 78 items where the two differ gives embeddings 61.9% of decided pairs, CI [0.384, 0.819],
p = 0.38. An 80%-powered test needs 137 decided pairs and only 78 items differ at all, so this
golden set can never settle it. The mechanism is right and the measurement is underpowered.

One example, since the aggregate hides it. On g027 (*"I updated iOS yesterday and ever since ... it
will randomly pause it"*) TF-IDF's nearest case was an iOS complaint about *missing songs* at 0.34
similarity, and the draft asked about missing songs. Dense retrieval found *"my app and phone are
up to date but the Spotify keeps pausing randomly"* at 0.92, and the draft asks for the iOS and
Spotify versions.

## 6. Top 5 failure modes

**1. Retrieval matches vocabulary, not the problem, and the draft inherits the wrong question.**
The dominant quality failure: 48 of 200 replies scored 2 or below from the judge, mostly this
shape. Customer (g027): *"I updated iOS yesterday and ever since every time I listen to music on
Spotify it will randomly pause it"*. Agent: *"Are your songs missing from Your Music section or
from a playlist?"* The nearest neighbours were iOS-update complaints about missing *songs*, and the
model copied their clarifying question. Same in g088 (app stuck on "Connecting", answered about
skipping) and g135 (Chromecast shuffle, answered about Chromecast discovery). Hypothesis: TF-IDF
similarity peaks at 0.26-0.39 here, far below the 0.6+ where neighbours are genuinely the same
issue, and the prompt tells the model to follow the exemplars. Partly confirmed: dense retrieval
finds the right precedent on all three (section 5), though I could not measure the reply gain.

**2. `other` is a high-prior sink that pulls tweets into needless escalation.** 13 of the 42
over-escalations landed in `other`. Three examples, with the confidence and the true intent:
*"I log in to Spotify for the first time in years and I get a 503 error"* (0.39, playback),
*"Okay I thought I was the only one. But Spotify is down"* (0.39, playback), *"how do I go about
buying annual membership please ??!!!"* (0.54, plan_and_offers).

Cause: `other` holds 11.8% of silver labels, spanning venting, jokes, artist questions and
support-process complaints, so it has a high prior and no coherent lexical signature. Fix: split
out `creator_or_business` and `support_process_complaint`, as I already did `praise_or_thanks`.

**3. Outage traffic is time-sensitive and the agent has no notion of "now".** The eval window
contains a real Spotify incident: 11 of 200 golden items are outage tweets. Spotify's own replies
are things like *"Everything should be running smoothly now"*, true at the minute sent and false if
replayed. The agent retrieves them as precedent with no way to check current status, and separately
misroutes several into `other`. I would fix this first in production: it is high volume, highly
correlated (one incident produces hundreds of near-identical tweets) and confidently wrong. Fix: a
status-page signal as a first-class input, plus an incident mode.

**4. Root-only modelling loses follow-up context, producing one of the two unsafe auto-replies.**
g095: *"Still nothing, even when I'm on WiFi. Any solution to this? Seems like a fatal flaw in your
login system."* It is a follow-up whose real issue is login, needing account access. Alone it
reads as connectivity, so the agent classified it `playback_technical` (0.43) and publicly asked
for iOS and app versions. The other unsafe reply, g165, is an album-recommendation request classified
`content_availability` and answered with a licensing link. Fix: use the thread-context columns
`data.py` already builds, and treat "still", "again" and "as I said" as continuation signals.

**5. Small classes collapse into their larger neighbours.** `device_integration` scores F1 0.40 on
67 silver examples, with 4 of 8 golden device tweets classified `playback_technical`. Consequence
(g081): *"I can't play my heavily curated Christmas playlist on the tv because you murdered the
Roku app"* got a generic feedback acknowledgement instead of Spotify's Roku answer. Same for
`praise_or_thanks` (F1 0.40, 49 examples). Cause: silver labels are sampled uniformly, so rare
intents stay rare. Fix: active sampling for the silver pass, targeting low-confidence and
keyword-matched examples for the small classes.

A sixth, smaller one: an abusive tweet was auto-answered with a polite request for detail. Not
wrong, but abuse should be its own route.

## 7. What is misleading about my headline number

The headline is "41.9% auto-handled at 0.970 precision, 2 unsafe replies against 19". Six reasons
to discount it.

**The 0.970 is 2 errors out of 67 auto-handled items, Wilson 95% CI [0.898, 0.992].** I can defend
"much safer than copy-the-nearest-case". I cannot defend the third decimal. The same interval means
the true unsafe rate could plausibly be 3x what I measured.

**I wrote the labels and built the system, with no second annotator.** Every escalation boundary
case was decided by the person who also chose the policy. The rule "a public reply is fine if a
help-centre answer exists" is mine, and a support lead might draw it elsewhere. My labels already
disagree with Spotify's own behaviour on 34 of 200 items. `LABELING.md` flags 16 items
where a second labeller could reasonably differ, enough to move auto precision by a couple of
points. Inter-annotator agreement is unknown, not high.

**Precision is measured only on what the agent chose to answer, which is the easy half.** It
conditions on the agent's own decision, so a system that answers only the easy tweets scores well.
Coverage has to be read beside it, which is also why the trivial baseline's zero errors is worth
nothing.

**Reply quality is the weakest evidence.** My human ratings cover 20 items per system, 60 total,
rated by me, and the trivial baseline is identifiable from its constant text, so blinding is
imperfect. The agent's mean human rating is 2.85 of 5: better than both baselines, still below
"good". The judge cannot substitute, since it ranks the systems in the wrong order.

**Two weeks of one brand in 2017, part of it an outage.** The eval window is 11% outage tweets,
which inflates the apparently-public share of traffic and understates difficulty, since
near-duplicates are easy to retrieve. Nothing here transfers to another brand, another period, or
a channel allowing more than 280 characters, and the `content_availability` numbers are partly a
Taylor Swift release-week artefact.

**The reply-quality metric rewards deleting the safety policy.** The variant that auto-answers
everything scores the *best* judge mean (3.67 against 3.52) while sending 59 unsafe replies instead
of 2. A headline built on reply quality is anti-correlated with what I claim to care about, over
exactly the change a team would be tempted to make. The two numbers must be read together.

Two claims survive all of that. The 39% wrong-name rate of the copy-paste baseline against 0% for
the agent is a count over all 200 items with no labelling judgement in it. And the paired
significance of the escalation advantage (p = 7.6e-05) depends only on the sign of the per-item
disagreements, not on my precision estimate being well calibrated.

One claim I am not making: that embedding retrieval improves reply quality. The examples suggest it
does, the blinded test cannot confirm it at 21 decided pairs, and the golden set is too small to
ever confirm an effect that size.

## 8. What I would do with one more week

1. **Build an evaluation set that can adjudicate retrieval.** The embedding swap is implemented and
   looks right, and this golden set provably cannot measure it (137 decided pairs needed, 78
   available). Fix: a few hundred items sampled where retrieval is weak, scored pairwise, plus a
   retrieval-only metric (does the top neighbour share the hand-labelled intent?) needing no reply
   rating at all.
2. **A second annotator on 80 items**, chosen to over-sample the boundary cases flagged in my
   notes, to put a real number on label reliability and therefore on auto precision.
3. **Incident awareness.** A status signal as an input, an incident intent, and a rule suppressing
   replayed "everything is fine now" precedents when current status is unknown.
4. **Retire the LLM judge for ranking, keep it for triage.** The pairwise harness is built
   (`supportagent.pairwise`). The next step is to run the *judge* through it as a pairwise judge
   and re-measure agreement, since absolute scoring is where it fails. Promote the deterministic
   checks (wrong name, invented link, false DM claim, promise) to blocking tests in CI, because
   they caught what the judge missed.
5. **Split `other`, and active-sample the silver pass** for the small classes, addressing failure
   modes 2 and 5 together.
6. **Report a business-shaped number.** Auto precision is an ML metric. A support lead needs
   deflected contacts per week against the expected cost of a wrong public reply, with escalation
   reasons attached so the queue arrives pre-triaged. That framing also makes the coverage and
   precision trade the lead's decision rather than mine.
