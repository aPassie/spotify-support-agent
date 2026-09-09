# Golden set: sampling and labelling protocol

**Unit.** One *root* customer tweet addressed to SpotifyCares (the first message of a thread)
from the held-out period **2017-11-20 .. 2017-12-03**, i.e. after every message used for
retrieval, silver labels and classifier training (`configs/agent.yaml: eval_start`).

**Sampling (200 items).**
- `random` (160): uniform random sample of eval-period roots, any language. Gives an
  unbiased estimate of the live traffic mix.
- `targeted:<intent>` (5 x 8 = 40): keyword-matched extras for intents that are rare in a
  uniform sample (ads, devices, downloads, content, feature requests) so every class has
  enough examples for a per-class number. Keyword matching means these items are *easier*
  than average for the keyword baseline; headline metrics are therefore reported on the
  `random` stratum only, with per-class numbers on the full set.

**Labels (per item).**
- `intent`: one of the 10 intents in `configs/intents.yaml`, chosen for the customer's *main*
  problem or request. Tie-breaks used: a premium user who hears ads or sees "Free" is
  `billing_payment` (state of the paid subscription), not `ads_complaint`; a song that exists
  but will not play is `playback_technical`, a missing/greyed/wrong song is
  `content_availability`; asking what a plan/offer includes or costs is `plan_and_offers`;
  complaints about recommendations, shuffle or removed features are `feature_request_feedback`.
- `should_escalate` (the decision a good agent should make):
  - **True** if any of: resolving it needs someone to look at or change the customer's account
    (billing, login, plan membership, offer eligibility for *this* account); a refund or charge
    dispute; personal data posted publicly; a possible account compromise; legal/safety content;
    not in English; the request lives in an image/link or is too vague to act on; or a
    complaint about the support process itself.
  - **False** if a grounded public reply resolves or properly advances the issue: a
    troubleshooting step, a help-centre link, a licensing/availability explanation, a feedback
    acknowledgement, factual plan/offer information that does not depend on the customer's
    account, a clarifying question about a technical issue, or a thank-you for praise.
  - The label is what *should* happen, not what SpotifyCares did. The historical reply was
    shown during labelling as context and is stored in `hist_brand_reply`, but in 14 of 200
    cases the label deliberately disagrees with it (mostly how-to billing/plan questions that
    SpotifyCares took to DM but that have a public help-centre answer).
- `escalation_reason`: primary reason id from `configs/intents.yaml` when `should_escalate`.
- `is_english`: whether the tweet is actually English (the automatic language id was wrong on
  8 of 200 items, all short English tweets flagged as Swedish/German/Dutch/Tagalog/Spanish).
- `notes`: free text on ambiguity.

**Who labelled.** All 200 were labelled by one person in a single pass with the protocol above,
then re-read once for consistency. There is no second annotator, so inter-annotator agreement
is unknown; the ambiguity notes mark the ~20 items where a second labeller could reasonably
differ (mostly `plan_and_offers` vs `billing_payment`, and `other` vs a real intent).
