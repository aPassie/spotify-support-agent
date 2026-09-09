# Golden set: sampling and labelling protocol

**Unit.** One *root* customer tweet addressed to SpotifyCares (the first message of a thread)
from the held-out period **2017-11-20 .. 2017-12-03**, i.e. after every message used for
retrieval, silver labels and classifier training (`configs/agent.yaml: eval_start`).

**Sampling (200 items).**
- `random` (160): uniform random sample of eval-period roots, any language. Gives an
  unbiased estimate of the live traffic mix.
- `targeted:<intent>` (5 x 8 = 40): keyword-matched extras for intents that are rare in a
  uniform sample (ads, devices, downloads, content, feature requests) so every class has
  enough examples for a per-class number. Keyword matching makes these items *easier*
  than average for the keyword baseline. Headline metrics are therefore reported on the
  `random` stratum only, with per-class numbers on the full set.

**Labels (per item).**
- `intent`: one of the 11 intents in `configs/intents.yaml`, for the customer's *main* problem or
  request. Tie-breaks used:
  - a premium user who hears ads or sees "Free" is `billing_payment`, the state of the paid
    subscription, not `ads_complaint`
  - a song that exists but will not play is `playback_technical`; a missing, greyed-out or wrong
    song is `content_availability`
  - asking what a plan or offer includes or costs is `plan_and_offers`
  - complaints about recommendations, shuffle or removed features are `feature_request_feedback`
- `should_escalate` (the decision a good agent should make). **True** when any of these hold:
  - resolving it needs someone to read or change this customer's account: billing, login, plan
    membership, or offer eligibility
  - a refund or charge dispute
  - personal data posted publicly, or a possible account compromise
  - legal or safety content
  - not in English
  - the request lives in an image or link, or is too vague to act on
  - a complaint about the support process itself

  **False** when a grounded public reply resolves or advances the issue: a troubleshooting step, a
  help-centre link, a licensing or availability explanation, a feedback acknowledgement, plan and
  offer facts that do not depend on this account, a clarifying question, or a thank-you.
  - The label is what *should* happen, not what SpotifyCares did. The historical reply was visible
    during labelling and is stored in `hist_brand_reply`. The label disagrees with it on 34 of 200
    items: 20 where Spotify moved to DM but a public help-centre answer exists, and 14 where
    Spotify answered publicly but the case needed account access.
- `escalation_reason`: primary reason id from `configs/intents.yaml` when `should_escalate`.
- `is_english`: whether the tweet is actually English (the automatic language id was wrong on
  8 of 200 items, all short English tweets flagged as Swedish/German/Dutch/Tagalog/Spanish).
- `notes`: free text on ambiguity.

**Who labelled.** One person, in a single pass with the protocol above, re-read once for
consistency. There is no second annotator, so inter-annotator agreement is unknown. The `notes`
field flags 16 items where a second labeller could reasonably differ, mostly `plan_and_offers`
against `billing_payment`, and `other` against a real intent.
