from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Churn Prediction API",
    description="API to predict customer churn based on RFM and behavioral data",
    version="1.0.0"
)

# Load the model on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Could not load model. {str(e)}")

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

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_risk: str
    action_recommended: str

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "RandomForestClassifier"}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")
    
    try:
        # Convert to DataFrame for the Pipeline
        data = pd.DataFrame([customer.dict()])
        
        # Predict Probability
        probs = model.predict_proba(data)
        churn_prob = float(probs[0][1])  # Probability of class 1 (Churn)
        
        # Threshold Logic (Using 0.45 from Part 3)
        if churn_prob >= 0.70:
            risk = "High"
            action = "Immediate high-value win-back offer / Concierge support"
        elif churn_prob >= 0.45:
            risk = "Medium"
            action = "Targeted discount / Engage via email"
        else:
            risk = "Low"
            action = "Standard loyalty nurturing"
            
        return PredictionResponse(
            churn_probability=round(churn_prob, 4),
            churn_risk=risk,
            action_recommended=action
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
