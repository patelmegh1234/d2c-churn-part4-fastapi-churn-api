# Part 4: FastAPI Churn Scoring Service

A production-ready REST API that serves a trained Random Forest churn-prediction model for a D2C personal-care brand. The CRM team can query individual or batches of customers to get churn probability, risk level, and a recommended action.

---

## Repository Structure

```
d2c-churn-part4-fastapi-churn-api/
├── main.py               # FastAPI application (all endpoints)
├── model.pkl             # Trained Random Forest pipeline (from Part 3)
├── test_main.py          # Pytest test suite (12 test cases)
├── monitoring_plan.md    # Post-deployment monitoring & responsible-use plan
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the API Server
```bash
uvicorn main:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

Interactive Swagger UI: `http://127.0.0.1:8000/docs`

---

## Endpoints

### `GET /health`
Returns the API and model health status.

**Sample Request:**
```bash
curl http://127.0.0.1:8000/health
```

**Sample Response:**
```json
{
  "status": "ok",
  "model": "RandomForestClassifier",
  "version": "1.0.0"
}
```

---

### `POST /predict`
Predicts churn risk for a **single** customer.

**Sample Request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city_tier": "Tier 1",
    "age_group": "25-34",
    "loyalty_tier": "Silver",
    "recency_days": 45.0,
    "frequency_180d": 3.0,
    "monetary_180d": 1500.0,
    "ticket_count_90d": 1.0,
    "sessions_30d": 5.0,
    "last_visit_days_ago": 15.0
  }'
```

**Sample Response:**
```json
{
  "churn_probability": 0.6123,
  "predicted_class": 1,
  "risk_level": "medium",
  "risk_explanation": "Churn probability is 61%. Moderate engagement drop detected. A targeted discount or re-engagement email campaign is recommended."
}
```

**Risk Levels:**
| Risk Level | Churn Probability | Recommended Action |
|---|---|---|
| `low` | < 0.45 | Standard loyalty nurturing |
| `medium` | 0.45 – 0.70 | Targeted discount / re-engagement email |
| `high` | ≥ 0.70 | Immediate win-back offer / Concierge support |

---

### `POST /batch_predict`
Predicts churn risk for **multiple** customers in one request (max 1000 per call).

**Sample Request:**
```bash
curl -X POST http://127.0.0.1:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '[
    {"customer_id": "CUST001", "recency_days": 10.0, "sessions_30d": 20.0, "frequency_180d": 8.0},
    {"customer_id": "CUST002", "recency_days": 200.0, "sessions_30d": 0.0, "ticket_count_90d": 4.0}
  ]'
```

**Sample Response:**
```json
[
  {
    "customer_id": "CUST001",
    "churn_probability": 0.1234,
    "predicted_class": 0,
    "risk_level": "low",
    "risk_explanation": "Churn probability is 12%. Customer shows healthy engagement signals. Standard loyalty nurturing is sufficient."
  },
  {
    "customer_id": "CUST002",
    "churn_probability": 0.8512,
    "predicted_class": 1,
    "risk_level": "high",
    "risk_explanation": "Churn probability is 85%. Low recent activity and elevated support-ticket count indicate high churn risk. Immediate concierge outreach or win-back offer is recommended."
  }
]
```

---

## Run Tests

```bash
pytest test_main.py -v
```

Expected output: **12 tests passed**

---

## Model Notes

- **Model type**: Scikit-Learn `Pipeline` (preprocessing + `RandomForestClassifier`)
- **Decision threshold**: 0.45 (tuned to maximize Recall for retention use cases)
- **Feature set**: 25 features based on RFM, support history, and web/app engagement — all computed from data available on or before the customer snapshot date (`2025-09-30`)
- **Source**: Model trained in [Part 3 repository](https://github.com/patelmegh1234/d2c-churn-part3-churn-model)

## Data Notes

The model was trained on a snapshot of customer data up to `2025-09-30`. Do not pass post-snapshot features as inputs. All feature values must represent the customer's state **at the time of scoring**, not future events.

## Monitoring & Responsible Use

See [`monitoring_plan.md`](monitoring_plan.md) for:
- Data drift detection strategy
- Prediction distribution monitoring
- Business outcome tracking
- API error alerting
- Retraining triggers
- Responsible use guidelines (who/how the API should and should NOT be used)
