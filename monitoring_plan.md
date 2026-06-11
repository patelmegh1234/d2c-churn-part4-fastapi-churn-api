# Monitoring Plan & Responsible Use

## 1. Overview

This document defines what should be tracked after the Churn Prediction API is deployed, and how the retention team should — and should **not** — use the predictions.

---

## 2. Monitoring Plan

### 2.1 Data Drift
| Signal | What to Track | Alert Threshold |
|---|---|---|
| Feature distribution | Weekly histogram comparison of key features (recency_days, sessions_30d, ticket_count_90d) vs. training baseline | KS-test p-value < 0.05 |
| Missing/null rates | % of API requests with null feature values per field | > 5% null rate on any key feature |
| Categorical drift | New unseen values in `loyalty_tier`, `city_tier`, `age_group` | Any new category not in training set |

### 2.2 Prediction Distribution
| Signal | What to Track | Alert Threshold |
|---|---|---|
| High-risk rate | % of customers scored as `high` risk per day | > 30% or < 5% (unusual shifts) |
| Score histogram | Daily distribution of `churn_probability` | Significant shift from training-time distribution |
| Predicted churn rate | Daily % of `predicted_class = 1` | > 2x the baseline train-set churn rate |

### 2.3 Business Outcomes
| Signal | What to Track | Cadence |
|---|---|---|
| Retention campaign response rate | Did `high`-risk customers respond to offers? | Monthly |
| Actual churn rate vs. predicted | Compare predictions made 60 days ago against actual churn outcomes | Monthly |
| ROI per segment | Revenue saved from targeted retention vs. cost of intervention | Quarterly |

### 2.4 API Health / Errors
| Signal | What to Track | Alert Threshold |
|---|---|---|
| HTTP 400/422 error rate | Malformed or invalid input | > 1% of requests |
| HTTP 503 errors | Model-load failure | Any occurrence |
| Latency (P95) | Time to return a prediction response | > 500ms |
| Uptime | Server availability | < 99.5% |

### 2.5 Retraining Triggers
The model should be retrained if **any** of the following occur:
- Actual churn rate deviates from predicted churn rate by more than **10 percentage points** for 2 consecutive months.
- A major product, pricing, or market event occurs (e.g., new product launch, competitor entry).
- A data-drift alert fires on more than 3 key features simultaneously.
- Model F1-score on a fresh holdout sample drops below **0.70**.
- The training snapshot date is more than **6 months** old.

---

## 3. Responsible Use Guidelines

### 3.1 How the Retention Team **Should** Use the API
- Use predictions as **one input** in a multi-factor decision about customer outreach. Do not rely on the score alone.
- Pair the `risk_level` with **customer LTV (lifetime value)** before deciding the size of a discount or intervention. A `high`-risk, low-LTV customer may not justify a large offer.
- Use `batch_predict` to run weekly sweeps of all active customers and prioritize the team's outreach queue.
- Treat a `medium` risk score as a prompt to investigate further — check recent tickets, return history, or session activity before acting.
- Document all interventions and their outcomes to build a feedback loop for model retraining.

### 3.2 How the Retention Team **Should NOT** Use the API
- **Do not** use the `predicted_class` as an automatic trigger to send discounts without human review for high-spend decisions.
- **Do not** use churn scores to deny service, downgrade account tiers, or treat customers negatively.
- **Do not** share raw churn probability scores with the customer — this could erode trust and brand perception.
- **Do not** target customers who have explicitly opted out of marketing (`marketing_consent = 0`) with promotional campaigns, regardless of their churn score.
- **Do not** use the model for decisions outside its intended scope (e.g., credit scoring, fraud detection).
- **Do not** assume a `low` risk score means a customer is completely safe. Low-risk customers who suddenly change behaviour should still be monitored via web-activity signals.

### 3.3 Fairness and Bias Considerations
- The model was trained on historical D2C data and may reflect historical biases (e.g., customers from lower-tier cities may have lower feature completeness).
- Segment-level audit: periodically check that false-positive and false-negative rates are consistent across `city_tier`, `age_group`, and `acquisition_channel` to avoid disparate impact.
- If a subgroup has a significantly higher false-negative rate (model misses their churn), those customers may be under-served by retention campaigns.
