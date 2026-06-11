import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ── Shared payload ────────────────────────────────────────────────────────────
VALID_PAYLOAD = {
    "city_tier": "Tier 1",
    "age_group": "25-34",
    "acquisition_channel": "Organic",
    "loyalty_tier": "Silver",
    "preferred_category": "Skin Care",
    "marketing_consent": 1,
    "recency_days": 45.0,
    "frequency_180d": 3.0,
    "monetary_180d": 1500.0,
    "return_rate_180d": 0.1,
    "avg_discount_pct_180d": 5.0,
    "avg_rating_180d": 3.8,
    "category_diversity_180d": 2.0,
    "ticket_count_90d": 1.0,
    "negative_ticket_rate_90d": 0.5,
    "avg_resolution_hours_90d": 24.0,
    "days_since_signup": 365.0,
    "sessions_30d": 5.0,
    "product_views_30d": 20.0,
    "cart_adds_30d": 2.0,
    "wishlist_adds_30d": 1.0,
    "abandoned_carts_30d": 1.0,
    "email_opens_30d": 2.0,
    "campaign_clicks_30d": 0.0,
    "last_visit_days_ago": 15.0,
}

HIGH_RISK_PAYLOAD = {
    "recency_days": 180.0,
    "frequency_180d": 0.0,
    "monetary_180d": 0.0,
    "sessions_30d": 0.0,
    "last_visit_days_ago": 90.0,
    "ticket_count_90d": 5.0,
    "abandoned_carts_30d": 3.0,
}

LOW_RISK_PAYLOAD = {
    "recency_days": 2.0,
    "frequency_180d": 15.0,
    "monetary_180d": 10000.0,
    "sessions_30d": 30.0,
    "last_visit_days_ago": 1.0,
    "ticket_count_90d": 0.0,
    "loyalty_tier": "Platinum",
}


# ── /health ───────────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model" in data

    def test_health_returns_version(self):
        response = client.get("/health")
        assert "version" in response.json()


# ── /predict ──────────────────────────────────────────────────────────────────
class TestPredict:
    def test_predict_valid_full_payload(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert "predicted_class" in data
        assert "risk_level" in data
        assert "risk_explanation" in data
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["predicted_class"] in [0, 1]
        assert data["risk_level"] in ["low", "medium", "high"]

    def test_predict_empty_payload_uses_defaults(self):
        """All fields have defaults so an empty dict should succeed."""
        response = client.post("/predict", json={})
        assert response.status_code == 200
        assert "churn_probability" in response.json()

    def test_predict_invalid_data_type_returns_422(self):
        response = client.post("/predict", json={"recency_days": "not_a_number"})
        assert response.status_code == 422

    def test_predict_high_risk_customer(self):
        """Dormant customer with many support tickets should get a high/medium risk."""
        response = client.post("/predict", json=HIGH_RISK_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_level"] in ["high", "medium"]

    def test_predict_low_risk_customer(self):
        """Very active loyal customer should get low/medium risk."""
        response = client.post("/predict", json=LOW_RISK_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_level"] in ["low", "medium"]

    def test_predict_probability_is_float(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert isinstance(response.json()["churn_probability"], float)


# ── /batch_predict ────────────────────────────────────────────────────────────
class TestBatchPredict:
    def test_batch_predict_multiple_customers(self):
        batch = [
            {"customer_id": "CUST001", **VALID_PAYLOAD},
            {"customer_id": "CUST002", **HIGH_RISK_PAYLOAD},
            {"customer_id": "CUST003", **LOW_RISK_PAYLOAD},
        ]
        response = client.post("/batch_predict", json=batch)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        for item in data:
            assert "churn_probability" in item
            assert "predicted_class" in item
            assert "risk_level" in item

    def test_batch_predict_single_customer(self):
        response = client.post("/batch_predict", json=[VALID_PAYLOAD])
        assert response.status_code == 200
        assert len(response.json()) == 1

    def test_batch_predict_preserves_customer_id(self):
        batch = [{"customer_id": "CUST_TEST_42", **VALID_PAYLOAD}]
        response = client.post("/batch_predict", json=batch)
        assert response.status_code == 200
        assert response.json()[0]["customer_id"] == "CUST_TEST_42"

    def test_batch_predict_empty_list_returns_400(self):
        response = client.post("/batch_predict", json=[])
        assert response.status_code == 400
