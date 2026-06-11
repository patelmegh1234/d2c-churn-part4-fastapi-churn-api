import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model": "RandomForestClassifier"}

def test_predict_endpoint_valid_data():
    payload = {
        "city_tier": "Tier 1",
        "age_group": "25-34",
        "recency_days": 10.0,
        "frequency_180d": 5.0,
        "monetary_180d": 5000.0,
        "sessions_30d": 15.0,
        "last_visit_days_ago": 2.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_risk" in data
    assert "action_recommended" in data
    assert 0.0 <= data["churn_probability"] <= 1.0

def test_predict_endpoint_missing_fields():
    # Because of Pydantic Optional fields with defaults, an empty dict is still valid!
    response = client.post("/predict", json={})
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data

def test_predict_endpoint_invalid_data_type():
    payload = {
        "recency_days": "not_a_number"
    }
    response = client.post("/predict", json=payload)
    # Should fail Pydantic validation
    assert response.status_code == 422
