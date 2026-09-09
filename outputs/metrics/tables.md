# Metrics (golden n=200, random stratum n=160)


## Intent classification (random)

| system | accuracy [95% CI] | macro-F1 |
|---|---|---|
| agent | 0.681 [0.606, 0.750] | 0.559 |
| trivial | 0.144 [0.094, 0.206] | 0.023 |
| simple | 0.494 [0.419, 0.569] | 0.472 |
| abl:embed | 0.681 [0.606, 0.750] | 0.559 |
| abl:no_exemplars | 0.681 [0.606, 0.750] | 0.559 |
| abl:no_policy | 0.681 [0.606, 0.750] | 0.559 |
| abl:rules_clf | 0.494 [0.419, 0.569] | 0.472 |

## Intent classification (all)

| system | accuracy [95% CI] | macro-F1 |
|---|---|---|
| agent | 0.665 [0.600, 0.730] | 0.627 |
| trivial | 0.165 [0.115, 0.220] | 0.026 |
| simple | 0.510 [0.440, 0.575] | 0.527 |
| abl:embed | 0.665 [0.600, 0.730] | 0.627 |
| abl:no_exemplars | 0.665 [0.600, 0.730] | 0.627 |
| abl:no_policy | 0.665 [0.600, 0.730] | 0.627 |
| abl:rules_clf | 0.510 [0.440, 0.575] | 0.527 |

## Escalation decision, random stratum with 95% CIs

`auto-handle rate` = share of tweets answered without a human. `auto precision` = of those, the share that should indeed have been answered publicly. `unsafe replies` = public replies sent on tweets that needed a human (the error that matters).

| system | auto-handle rate [CI] | auto precision [CI] | unsafe replies | escalate recall |
|---|---|---|---|---|
| agent | 0.419 [0.344, 0.500] | 0.970 [0.898, 0.992] | 2 of 67 | 0.966 |
| trivial | 0.000 [0.000, 0.000] | n/a | 0 of 0 | 1.000 |
| simple | 0.625 [0.556, 0.700] | 0.810 [0.722, 0.875] | 19 of 100 | 0.678 |
| abl:embed | 0.438 [0.362, 0.519] | 0.971 [0.902, 0.992] | 2 of 70 | 0.966 |
| abl:no_exemplars | 0.419 [0.344, 0.500] | 0.970 [0.898, 0.992] | 2 of 67 | 0.966 |
| abl:no_policy | 1.000 [1.000, 1.000] | 0.631 [0.554, 0.702] | 59 of 160 | 0.000 |
| abl:rules_clf | 0.263 [0.194, 0.331] | 0.929 [0.810, 0.975] | 3 of 42 | 0.949 |

### Paired significance vs the agent (McNemar exact, random stratum)

Same 160 items for every system, so the comparison is paired. `discordant` counts items where exactly one of the two systems erred.

| comparison | agent-only errors | other-only errors | discordant | p |
|---|---|---|---|---|
| vs trivial (unsafe replies) | 2 | 0 | 2 | 0.5 |
| vs trivial (intent errors) | 7 | 93 | 100 | 2.7e-20 |
| vs simple (unsafe replies) | 1 | 18 | 19 | 7.6e-05 |
| vs simple (intent errors) | 15 | 45 | 60 | 0.00013 |
| vs abl:embed (unsafe replies) | 0 | 0 | 0 | 1 |
| vs abl:embed (intent errors) | 0 | 0 | 0 | 1 |
| vs abl:no_exemplars (unsafe replies) | 0 | 0 | 0 | 1 |
| vs abl:no_exemplars (intent errors) | 0 | 0 | 0 | 1 |
| vs abl:no_policy (unsafe replies) | 0 | 57 | 57 | 1.4e-17 |
| vs abl:no_policy (intent errors) | 0 | 0 | 0 | 1 |
| vs abl:rules_clf (unsafe replies) | 1 | 2 | 3 | 1 |
| vs abl:rules_clf (intent errors) | 15 | 45 | 60 | 0.00013 |

## Escalation decision (all 200)

| system | auto-handle rate | auto precision | unsafe auto rate | escalate precision | escalate recall |
|---|---|---|---|---|---|
| agent | 0.460 | 0.978 | 0.029 (2) | 0.611 | 0.971 |
| trivial | 0.000 | 0.000 | 0.000 (0) | 0.340 | 1.000 |
| simple | 0.635 | 0.835 | 0.309 (21) | 0.644 | 0.691 |
| abl:embed | 0.470 | 0.968 | 0.044 (3) | 0.613 | 0.956 |
| abl:no_exemplars | 0.460 | 0.978 | 0.029 (2) | 0.611 | 0.971 |
| abl:no_policy | 1.000 | 0.660 | 1.000 (68) | 0.000 | 0.000 |
| abl:rules_clf | 0.335 | 0.940 | 0.059 (4) | 0.481 | 0.941 |

## Reply quality, LLM judge (all 200)

| system | mean overall [95% CI] | % >=4 | % <=2 | addresses | brand-consistent | actionable | safe | tone |
|---|---|---|---|---|---|---|---|---|
| agent | 3.52 [3.38, 3.66] | 0.59 | 0.24 | 0.81 | 0.61 | 0.81 | 0.96 | 0.98 |
| trivial | 2.81 [2.67, 2.96] | 0.33 | 0.58 | 0.57 | 0.34 | 0.50 | 0.65 | 0.72 |
| simple | 3.61 [3.44, 3.77] | 0.62 | 0.28 | 0.90 | 0.66 | 0.81 | 0.96 | 0.97 |
| abl:embed | 3.46 [3.31, 3.59] | 0.56 | 0.28 | 0.79 | 0.57 | 0.80 | 0.95 | 0.98 |
| abl:no_exemplars | 3.24 [3.11, 3.37] | 0.47 | 0.30 | 0.79 | 0.48 | 0.77 | 0.93 | 0.97 |
| abl:no_policy | 3.67 [3.54, 3.79] | 0.64 | 0.14 | 0.91 | 0.64 | 0.88 | 0.99 | 0.99 |
| abl:rules_clf | 3.38 [3.23, 3.52] | 0.54 | 0.32 | 0.73 | 0.55 | 0.73 | 0.96 | 0.99 |

### Judge score by hand label (all 200)

| system | label=auto-ok mean (n) | label=escalate mean (n) | system auto mean (n) | system escalate mean (n) |
|---|---|---|---|---|
| agent | 3.41 (132) | 3.75 (68) | 3.74 (92) | 3.34 (108) |
| trivial | 2.32 (132) | 3.78 (68) | nan (0) | 2.81 (200) |
| simple | 3.45 (132) | 3.91 (68) | 3.53 (127) | 3.75 (73) |
| abl:embed | 3.31 (132) | 3.74 (68) | 3.64 (94) | 3.29 (106) |
| abl:no_exemplars | 2.97 (132) | 3.76 (68) | 3.12 (92) | 3.34 (108) |
| abl:no_policy | 3.64 (132) | 3.72 (68) | 3.67 (200) | nan (0) |
| abl:rules_clf | 3.22 (132) | 3.68 (68) | 3.85 (67) | 3.14 (133) |

## Reply surface checks (all 200)

| system | mean chars | % >280 | % with link | % link not in history | % greet customer by a (necessarily wrong) name | ROUGE-L vs actual reply |
|---|---|---|---|---|---|---|
| agent | 135 | 0.00 | 0.52 | 0.00 | 0.00 | 0.384 |
| trivial | 132 | 0.00 | 1.00 | 0.00 | 0.00 | 0.335 |
| simple | 119 | 0.00 | 0.49 | 0.00 | 0.39 | 0.305 |
| abl:embed | 135 | 0.00 | 0.56 | 0.00 | 0.01 | 0.386 |
| abl:no_exemplars | 158 | 0.00 | 0.38 | 0.00 | 0.00 | 0.310 |
| abl:no_policy | 149 | 0.00 | 0.46 | 0.00 | 0.01 | 0.334 |
| abl:rules_clf | 128 | 0.00 | 0.44 | 0.00 | 0.00 | 0.368 |

## Judge vs human agreement

| subset | n | Spearman | weighted kappa | exact | within 1 | judge mean | human mean |
|---|---|---|---|---|---|---|---|
| all | 60 | 0.49 | 0.32 | 0.23 | 0.60 | 3.45 | 2.43 |
| trivial | 20 | 0.50 | 0.31 | 0.45 | 0.60 | 3.05 | 2.30 |
| simple | 20 | 0.59 | 0.31 | 0.10 | 0.60 | 3.60 | 2.15 |
| agent | 20 | 0.40 | 0.31 | 0.15 | 0.60 | 3.70 | 2.85 |

## Agent: per-class F1 (all 200)

| intent | F1 | n |
|---|---|---|
| account_access | 0.92 | 18 |
| billing_payment | 0.77 | 17 |
| plan_and_offers | 0.76 | 35 |
| playback_technical | 0.65 | 32 |
| downloads_offline | 0.60 | 7 |
| content_availability | 0.63 | 27 |
| feature_request_feedback | 0.61 | 33 |
| ads_complaint | 0.67 | 3 |
| device_integration | 0.40 | 8 |
| praise_or_thanks | 0.40 | 4 |
| other | 0.49 | 16 |

## Agent: primary escalation reasons

| reason | n |
|---|---|
| needs_account_access | 51 |
| unclear_request | 25 |
| historical_dm_pattern | 9 |
| no_precedent | 7 |
| language_routing | 7 |
| payment_dispute | 5 |
| pii_in_public | 2 |
| low_confidence | 1 |
| security_incident | 1 |

## Agent: guardrail flags on drafts

| flag | n |
|---|---|
| no_draft | 7 |
| removed_handle | 3 |
| too_long | 2 |
| sensitive_ask | 1 |
