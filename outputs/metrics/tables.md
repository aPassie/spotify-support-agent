# Metrics (golden n=200, random stratum n=160)


## Intent classification (random)

| system | accuracy | macro-F1 |
|---|---|---|
| agent | 0.681 | 0.559 |
| trivial | 0.144 | 0.023 |
| simple | 0.494 | 0.472 |

## Intent classification (all)

| system | accuracy | macro-F1 |
|---|---|---|
| agent | 0.665 | 0.627 |
| trivial | 0.165 | 0.026 |
| simple | 0.510 | 0.527 |

## Escalation decision (random stratum)

| system | auto-handle rate | auto precision | unsafe auto rate | escalate precision | escalate recall |
|---|---|---|---|---|---|
| agent | 0.419 | 0.970 | 0.034 (2) | 0.613 | 0.966 |
| trivial | 0.000 | 0.000 | 0.000 (0) | 0.369 | 1.000 |
| simple | 0.625 | 0.810 | 0.322 (19) | 0.667 | 0.678 |

## Escalation decision (all 200)

| system | auto-handle rate | auto precision | unsafe auto rate | escalate precision | escalate recall |
|---|---|---|---|---|---|
| agent | 0.460 | 0.978 | 0.029 (2) | 0.611 | 0.971 |
| trivial | 0.000 | 0.000 | 0.000 (0) | 0.340 | 1.000 |
| simple | 0.635 | 0.835 | 0.309 (21) | 0.644 | 0.691 |

## Reply quality, LLM judge (all 200)

| system | mean overall | % >=4 | % <=2 | addresses | brand-consistent | actionable | safe | tone |
|---|---|---|---|---|---|---|---|---|
| agent | 3.52 | 0.59 | 0.24 | 0.81 | 0.61 | 0.81 | 0.96 | 0.98 |
| trivial | 2.81 | 0.33 | 0.58 | 0.57 | 0.34 | 0.50 | 0.65 | 0.72 |
| simple | 3.61 | 0.62 | 0.28 | 0.90 | 0.66 | 0.81 | 0.96 | 0.97 |

### Judge score by hand label (all 200)

| system | label=auto-ok mean (n) | label=escalate mean (n) | system auto mean (n) | system escalate mean (n) |
|---|---|---|---|---|
| agent | 3.41 (132) | 3.75 (68) | 3.74 (92) | 3.34 (108) |
| trivial | 2.32 (132) | 3.78 (68) | nan (0) | 2.81 (200) |
| simple | 3.45 (132) | 3.91 (68) | 3.53 (127) | 3.75 (73) |

## Reply surface checks (all 200)

| system | mean chars | % >280 | % with link | % link not in history | % greet customer by a (necessarily wrong) name | ROUGE-L vs actual reply |
|---|---|---|---|---|---|---|
| agent | 135 | 0.00 | 0.52 | 0.00 | 0.00 | 0.384 |
| trivial | 132 | 0.00 | 1.00 | 0.00 | 0.00 | 0.335 |
| simple | 119 | 0.00 | 0.49 | 0.00 | 0.39 | 0.305 |

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
