# Part 4: FastAPI Churn Prediction API

This repository exposes the trained Churn Prediction Model from Part 3 as a REST API.

## Contents
- `main.py`: The FastAPI application defining the `/predict` endpoint and Pydantic schemas.
- `test_main.py`: Pytest file covering edge cases, valid inputs, missing values, and model errors.
- `model.pkl`: The trained Random Forest pipeline copied from Part 3.
- `requirements.txt`: Python dependencies.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
3. Test the endpoints:
   - Navigate to `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

## Run Tests
```bash
pytest test_main.py -v
```
