from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="D2C Churn Prediction API",
    description="""
    ## Churn Prediction Service
    
    This API serves a trained Random Forest churn-prediction model for a D2C personal-care brand.
    It exposes endpoints to predict the likelihood of a customer churning in the next 60 days,
    based on RFM metrics, support signals, and web engagement features.

    ### Endpoints:
    - **GET /health** – Health check
    - **POST /predict** – Predict churn for a single customer
    - **POST /batch_predict** – Predict churn for multiple customers at once
    """,
    version="1.0.0"
)

# Load the model on startup
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"WARNING: Could not load model: {str(e)}")


class CustomerFeatures(BaseModel):
    city_tier: Optional[str] = "Tier 1"
    age_group: Optional[str] = "25-34"
    acquisition_channel: Optional[str] = "Organic"
    loyalty_tier: Optional[str] = "No Tier"
    preferred_category: Optional[str] = "Unknown"
    marketing_consent: Optional[int] = 1
    recency_days: Optional[float] = 30.0
    frequency_180d: Optional[float] = 1.0
    monetary_180d: Optional[float] = 100.0
    return_rate_180d: Optional[float] = 0.0
    avg_discount_pct_180d: Optional[float] = 0.0
    avg_rating_180d: Optional[float] = 4.0
    category_diversity_180d: Optional[float] = 1.0
    ticket_count_90d: Optional[float] = 0.0
    negative_ticket_rate_90d: Optional[float] = 0.0
    avg_resolution_hours_90d: Optional[float] = 0.0
    days_since_signup: Optional[float] = 180.0
    sessions_30d: Optional[float] = 2.0
    product_views_30d: Optional[float] = 10.0
    cart_adds_30d: Optional[float] = 1.0
    wishlist_adds_30d: Optional[float] = 0.0
    abandoned_carts_30d: Optional[float] = 0.0
    email_opens_30d: Optional[float] = 1.0
    campaign_clicks_30d: Optional[float] = 0.0
    last_visit_days_ago: Optional[float] = 5.0

    model_config = {"json_schema_extra": {"example": {
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
                "last_visit_days_ago": 15.0
            }}}


class CustomerFeaturesWithId(CustomerFeatures):
    customer_id: Optional[str] = None


class PredictionResponse(BaseModel):
    churn_probability: float
    predicted_class: int
    risk_level: str
    risk_explanation: str


class BatchPredictionResponse(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    predicted_class: int
    risk_level: str
    risk_explanation: str


def _make_prediction(features_dict: dict) -> dict:
    """Core inference logic shared by /predict and /batch_predict."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable. Please check server logs.")

    # Remove customer_id if present – it is not a model feature
    features_dict.pop("customer_id", None)

    data = pd.DataFrame([features_dict])
    probs = model.predict_proba(data)
    churn_prob = float(probs[0][1])
    predicted_class = int(churn_prob >= 0.45)

    if churn_prob >= 0.70:
        risk_level = "high"
        explanation = (
            f"Churn probability is {churn_prob:.0%}. "
            "Low recent activity and elevated support-ticket count indicate high churn risk. "
            "Immediate concierge outreach or win-back offer is recommended."
        )
    elif churn_prob >= 0.45:
        risk_level = "medium"
        explanation = (
            f"Churn probability is {churn_prob:.0%}. "
            "Moderate engagement drop detected. "
            "A targeted discount or re-engagement email campaign is recommended."
        )
    else:
        risk_level = "low"
        explanation = (
            f"Churn probability is {churn_prob:.0%}. "
            "Customer shows healthy engagement signals. "
            "Standard loyalty nurturing is sufficient."
        )

    return {
        "churn_probability": round(churn_prob, 4),
        "predicted_class": predicted_class,
        "risk_level": risk_level,
        "risk_explanation": explanation,
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Returns API and model health status."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": "RandomForestClassifier", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerFeatures):
    """
    Predict churn risk for a **single** customer.

    Returns:
    - **churn_probability**: Float between 0 and 1
    - **predicted_class**: 1 = likely to churn, 0 = likely to stay
    - **risk_level**: `low`, `medium`, or `high`
    - **risk_explanation**: Plain-language rationale for the prediction
    """
    try:
        result = _make_prediction(customer.model_dump())
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=List[BatchPredictionResponse], tags=["Prediction"])
def batch_predict_churn(customers: List[CustomerFeaturesWithId]):
    """
    Predict churn risk for a **batch** of customers.

    Accepts a list of customer feature objects (each may optionally include a `customer_id`).

    Returns a list of prediction objects in the same order as the input.
    """
    if not customers:
        raise HTTPException(status_code=400, detail="Input list is empty. Provide at least one customer.")
    if len(customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 1000 customers per request.")

    results = []
    for customer in customers:
        customer_id = customer.model_dump().get("customer_id")
        try:
            result = _make_prediction(customer.model_dump())
            result["customer_id"] = customer_id
            results.append(BatchPredictionResponse(**result))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Prediction failed for customer {customer_id}: {str(e)}")

    return results
