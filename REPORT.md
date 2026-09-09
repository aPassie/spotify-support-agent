# SpotifyCares support agent: what I built, what it is worth, and where it breaks

## 1. Problem framing: what "good" means for this brand

I picked **SpotifyCares** by profiling the 40 highest-volume brands in the dataset on four
things: volume, how often the brand's reply asks the customer to move to DM, how often it
contains a link, and how often it is the first turn of a thread. Spotify sits in a useful middle:
26,000 first-turn exchanges, and a **37% DM rate**. Amazon and Hulu almost never move to DM
(nothing to learn about escalation), T-Mobile and Comcast almost always do (nothing to automate).
A brand that does both teaches both halves of the job.

Spotify support on Twitter is not a resolution channel; it is a **triage and deflection** channel.
Reading a few hundred exchanges, a good public reply is one of five things: a troubleshooting step,
a help-centre link, an explanation of why something is the way it is (licensing, plan limits), an
acknowledgement that routes feedback onward, or a clean hand-off to DM. So "good" here is:

- **Never send a public reply where a human was needed.** A wrong public reply on a billing or
  login problem is worse than silence: it is public, it is quotable, and it burns the customer's
  second attempt.
- **Deflect the genuinely public issues**, which are the majority of the volume.
- **Say why** when escalating, so the human picking it up starts ahead.

That ordering makes this a **precision problem, not a coverage problem**, and it is the reason my
headline metric is "of the tweets we auto-answered, how many should we have" rather than "how many
did we auto-answer".

### What I chose not to build

- **No multi-turn dialogue.** I model the *root* tweet of a thread and the brand's first public
  reply. Later turns in this dataset are mostly "we've replied to your DM", so the interesting
  content moves to a private channel the dataset does not contain. Modelling it would mean
  modelling text I cannot see.
- **No fine-tuning.** With 19.5k grounded exchanges available at inference time, retrieval buys
  the same brand-voice grounding as a LoRA, stays inspectable ("here are the five cases this reply
  came from"), and updates the moment the brand changes its help links.
- **No LLM in the classification loop.** The LLM labels 3,000 historical tweets once; a TF-IDF
  and logistic-regression classifier learns from those labels and then runs in microseconds with a
  probability the policy can threshold. The LLM is used only where it earns its cost: drafting.
- **No agent initials.** SpotifyCares signs 97% of replies with a human's initials ("/JN"). The
  agent never does. Imitating a named human is the one stylistic feature the brand should not want
  copied, and it would make the automation undisclosed.
- **No sentiment model.** Anger is not the trigger for escalation in this data; account state is.
  An angry playback complaint is still publicly answerable, and a polite billing question is not.

## 2. The system

```
tweet -> language id + PII scan
      -> intent classifier (TF-IDF word+char -> logistic regression, 11 classes)
      -> retrieve 8 nearest past customer tweets, with the brand's actual replies
      -> policy: auto or escalate, with reasons
      -> if auto: LLM drafts a reply from 4 exemplars, then hard guardrails
         if escalate: one of Spotify's own hand-off templates, LLM draft attached as an
                      internal suggestion for the human
```

**The intent taxonomy came from the data.** I clustered the 24,000 English root tweets (TF-IDF, LSA to
120 dims, k-means) and read samples from every cluster. Eleven intents survived. The important
part is that clustering also gave me the escalation policy for free: the brand's own DM rate per
cluster is bimodal, and the silver labels confirm it on 3,000 tweets.

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

Escalation also fires on evidence independent of the intent: PII in a public tweet, legal or
safety language, a possible account compromise, a refund demand, a non-English message, a
media-only tweet, no sufficiently similar precedent, or a majority of similar past cases having
gone to DM. Every escalation carries one or more stable reason ids, so I can measure *why* the
system escalates and not just how often.

**Guardrails, after generation.** A link that does not appear in the retrieved exemplars is
stripped (0 of 200 replies contained a link the brand had never used). Handles and initials are
removed. Replies over 280 characters are cut at a sentence boundary. A promise of a refund or fix,
or a request for a password or card details, flips the decision to escalate.

## 3. The golden set

200 root tweets from **2017-11-20 to 2017-12-03**, strictly after every tweet used for retrieval,
silver labels or training. 160 are a uniform random sample; 40 are keyword-targeted extras so
that rare intents have a usable count. Because the targeted items are easier for a keyword
baseline, **all headline numbers are on the 160-item random stratum**.

Each item has an intent, a should-escalate decision, a primary escalation reason, whether it is
really English, and a free-text ambiguity note. The decision label is *what should happen*, not
what Spotify did: in 34 of 200 cases they disagree, mostly how-to billing and plan questions that
Spotify took to DM but that have a public help-centre answer. Copying Spotify's own behaviour
would have made the nearest-neighbour baseline look artificially strong. Protocol, tie-break
rules and the single-annotator caveat are in `golden/LABELING.md`.

## 4. Results

Two baselines. **Trivial**: always escalate, majority intent, and the single most common Spotify
template (the "DM us your account email" message, used 7,351 times in the data). **Simple**:
keyword-rule intent, escalate if the single nearest past case went to DM, and reply with that
nearest historical reply verbatim.

### The decision that matters (160 random held-out tweets)

| system | auto-handle rate | auto precision | replies sent that needed a human | escalate recall |
|---|---|---|---|---|
| trivial | 0.0% | n/a | 0 | 1.000 |
| simple | 62.5% | 0.810 | **19** | 0.678 |
| **agent** | **41.9%** | **0.970** | **2** | **0.966** |

The simple baseline auto-answers half again as much traffic and gets 19 of those wrong. The agent
gives up a third of that coverage to cut the errors to 2. Given that a wrong public reply on a
billing problem costs more than a delayed correct one, that is the trade I would ship. The trivial
baseline is safe by construction and worth zero: it deflects nothing.

### Intent classification

| system | accuracy (random) | macro-F1 (random) |
|---|---|---|
| trivial (majority class) | 0.144 | 0.023 |
| simple (keyword rules) | 0.494 | 0.472 |
| agent (TF-IDF + LR on silver labels) | **0.681** | **0.559** |

Per class on all 200: `account_access` 0.92, `billing_payment` 0.77, `plan_and_offers` 0.76,
`ads_complaint` 0.67, `playback_technical` 0.65, `content_availability` 0.63,
`feature_request_feedback` 0.61, `downloads_offline` 0.60, `other` 0.49, `device_integration` 0.40,
`praise_or_thanks` 0.40. The three escalate-critical classes are the three strongest, which is the
right shape for this policy; the weak classes are all inside the auto-handle group where a
confusion costs a mediocre reply rather than an unsafe one.

### Reply quality: the LLM judge, and why I do not trust its ranking

The judge (gemma-3-4b-it, a different family from the Qwen generator) scores five binary checks
and a 1-5 overall, having been shown the customer tweet, the candidate reply, Spotify's actual
reply, and three similar past exchanges.

| system | judge mean | judge % >= 4 | my mean (blind, n=20 each) |
|---|---|---|---|
| trivial | 2.81 | 0.33 | 2.30 |
| simple | **3.61** | **0.62** | **2.15** |
| agent | 3.52 | 0.59 | **2.85** |

**The judge puts the copy-paste baseline first; I put it last.** That inversion is the most
important evaluation result in this project, and here is the mechanism. The judge sees the real
historical reply as a reference, and the simple baseline *is* a real historical reply, so it
matches the reference's style perfectly. What the judge does not notice is that the reply belongs
to a different customer. A deterministic check finds it:

| system | greets the customer by a name the system cannot possibly know |
|---|---|
| agent | **0%** |
| trivial | 0% |
| simple | **39%** (78 of 200) |

The dataset is anonymised, so no reply can legitimately know a customer's first name. Every one of
those 78 is Spotify greeting the wrong person by name in public. The judge rated `safe` at 0.96
for the simple baseline anyway. It also missed replies claiming "we've just replied to your DM"
when no DM existed, and a reply telling a listener to "reach out to your distributor".

Judge-human agreement over 60 blind ratings: Spearman 0.49, quadratic-weighted kappa 0.32, exact
agreement 0.23, within-one 0.60, and the judge is **+1.02 points more generous** on average. So the
judge is usable as a coarse filter (it does separate the trivial baseline from the other two) and
unusable as a ranker between systems of similar surface quality. Everything I actually claim about
reply quality rests on the 60 human ratings and the deterministic checks, not on the judge mean.

## 5. Top 5 failure modes

**1. Retrieval matches vocabulary, not the problem, and the draft inherits the wrong question.**
This is the dominant quality failure: 48 of 200 replies scored 2 or below from the judge, and most
look like this. Customer (g027): *"I updated iOS yesterday and ever since every time I listen to
music on Spotify it will randomly pause it, any idea what's going on?"* Agent: *"Are your songs
missing from Your Music section or from a playlist?"* The nearest neighbours were iOS-update
complaints about missing songs, and the model copied their clarifying question instead of answering
about pausing. Same shape in g088 (app stuck on "Connecting", answered about skipping) and g135
(Chromecast shuffle, answered about Chromecast discovery). Hypothesis: TF-IDF similarity peaks at
0.26-0.39 for these, well below the 0.6+ where neighbours are genuinely the same issue, and the
prompt tells the model to follow the exemplars. Fix: an embedding retriever, and a
similarity-weighted instruction that tells the model to answer directly when the top neighbour is
weak rather than mimicking it.

**2. `other` is a high-prior sink that pulls tweets into needless escalation.** 13 of the 42
over-escalations are tweets that landed in `other`. Examples: *"Wow... I log in to Spotify for the
first time in years and I get a 503 error"* (0.39 confidence, truly playback), *"Okay I thought I
was the only one. But Spotify is down"* (0.39, truly playback), *"how do I go about buying annual
membership please ??!!!"* (0.54, truly plan_and_offers). Cause: `other` holds 11.8% of silver labels and spans venting, jokes, artist questions and support-process complaints, so it has both a
high prior and no coherent lexical signature. Fix: split it into `praise_or_thanks` (already done,
which is where the class came from), `creator_or_business`, and `support_process_complaint`, and
leave a genuinely small residual.

**3. Outage traffic is time-sensitive and the agent has no notion of "now".** The eval window
contains a real Spotify incident: 11 of my 200 golden items are outage tweets, and 27 eval-period
tweets ask whether Spotify is down. Spotify's own replies are things like *"Everything should be
running smoothly now"* - true at the minute they were sent and false if replayed. The agent
retrieves them as precedent with no way to check current status, and separately misroutes several
into `other`. This is the failure I would fix first in production, because it is high volume,
highly correlated (one incident produces hundreds of near-identical tweets), and the wrong answer
is confidently wrong. Fix: a status-page signal as a first-class input, and an incident mode.

**4. Root-only modelling loses the context of follow-up tweets, which produced one of the two
unsafe auto-replies.** g095: *"Still nothing, even when I'm on WiFi. Any solution to this? Seems
like a fatal flaw in your login system."* This is a follow-up; the underlying issue is login, which
needs account access. Seen alone it reads as a connectivity problem, so the agent classified it
`playback_technical` (0.43) and publicly asked for iOS and app versions. The second unsafe reply,
g165, is a request for an album recommendation that was classified `content_availability` and
answered with a licensing link, which is nonsense. Fix: use the thread context columns that
`data.py` already builds, and treat "still", "again" and "as I said" as thread-continuation signals.

**5. Small classes collapse into their larger neighbours.** `device_integration` scores F1 0.40 with
67 silver examples, and 4 of 8 golden device tweets were classified `playback_technical`.
Consequence (g081): *"I can't play my heavily curated Christmas playlist on the tv because you
murdered the Roku app"* got a generic feedback acknowledgement instead of the Roku answer Spotify
actually gives. Same for `praise_or_thanks` (F1 0.40, 49 silver examples). Cause: silver labels are
sampled uniformly, so rare intents stay rare. Fix: active sampling for the silver pass, targeting
low-confidence and keyword-matched examples for the small classes.

Honourable mention: an abusive tweet (*"fix your app cunts"*) was auto-answered with a polite
request for detail. It is not wrong, but abuse should probably be its own route.

## 6. What is misleading about my headline number

The headline is "41.9% auto-handled at 0.970 precision, 2 unsafe replies against 19". Five reasons
to discount it.

**The 0.970 is 2 errors out of 67 auto-handled items, so its Wilson 95% confidence interval runs
from 0.898 to 0.992.** With n=160 and one annotator, I can defend "the agent is much safer than
copy-the-nearest-case" and I cannot defend the third decimal place. The same interval means the
true unsafe rate could plausibly be 3x what I measured.

**I wrote the labels and I built the system, and there is no second annotator.** Every escalation
boundary case was decided by the person who also chose the policy. The rule "a public reply is fine
if a help-centre answer exists" is mine, and a Spotify support lead might draw it elsewhere; note
that my labels already disagree with Spotify's own behaviour on 34 of 200 items. `LABELING.md`
marks roughly 20 items where a second labeller could reasonably differ, which is enough to move
auto precision by a couple of points. Inter-annotator agreement is unknown, not high.

**Precision is measured only on what the agent chose to answer, which is the easy half.** Auto
precision conditions on the agent's own decision, so a system that answers only trivially easy
tweets scores beautifully. That is exactly why the coverage column has to be read next to it, and
why the trivial baseline's "no errors" is worth nothing.

**Reply quality is the weakest part of the evidence.** My human ratings cover 20 items per system,
60 in total, rated by me, and the trivial baseline is identifiable from its constant text, so the
blinding is imperfect. The agent's *mean* human rating is 2.85 out of 5: better than both
baselines, and still below "good". The judge cannot substitute, because it ranks the systems in the
wrong order. The honest summary is that the agent's replies are noticeably better than the
alternatives and are not yet good.

**Two weeks of one brand in 2017, and part of it was an outage.** The eval window is 11%
outage tweets, which both inflates the apparently-public share of traffic and understates the
difficulty (near-duplicate tweets are easy to retrieve for). Nothing here says anything about
another brand, another period, or a channel where customers write more than 280 characters. The
`content_availability` numbers in particular are partly a Taylor Swift release-week artefact.

One thing I will defend without hedging: the 39% wrong-name rate of the copy-paste baseline, and
the 0% for the agent. That is a deterministic count over all 200 items with no labelling judgement
in it, and it is the clearest reason to prefer generation-with-guardrails over template reuse.

## 7. What I would do with one more week

1. **Swap TF-IDF retrieval for embeddings** and gate the draft on retrieval quality. Failure mode 1
   is the largest single quality problem and is unblocked by a 100 MB sentence-transformer.
2. **A second annotator on 80 items**, chosen to over-sample the boundary cases flagged in my
   notes, to put a real number on label reliability and therefore on auto precision.
3. **Incident awareness.** A status signal as an input, an incident intent, and a rule that
   suppresses replayed "everything is fine now" precedents when current status is unknown.
4. **Retire the LLM judge for ranking, keep it for triage.** Replace the mean-score claim with a
   pairwise A/B judge, which is far more robust than absolute scoring, and validate that against a
   larger set of human pairwise preferences. Add the deterministic checks (wrong name, invented
   link, false DM claim, promise) to the harness as blocking tests, since they caught what the
   judge missed.
5. **Split `other`, and active-sample the silver pass** for `device_integration` and the other
   small classes, addressing failure modes 2 and 5 together.
6. **Report a business-shaped number.** Right now the metric is auto precision; what a support lead
   needs is deflected contacts per week against expected cost of a wrong public reply, with the
   escalation reasons attached so the queue is pre-triaged. That framing also makes the coverage
   and precision trade explicit rather than my choosing it for them.
