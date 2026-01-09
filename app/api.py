from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(title="Credit Card Fraud Detection API")

# Global model variable
model = None



# Input schema (VERY IMPORTANT)

class Transaction(BaseModel):
    step: int = Field(..., ge=0)
    type: str
    amount: float = Field(..., gt=0)
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int = Field(0, ge=0, le=1)



# Load model on startup

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("fraud_pipeline.joblib")


@app.get("/")
def home():
    return {"status": "Credit Card Fraud Detection API is running"}


# Prediction endpoint

@app.post("/predict")
def predict_transaction(txn: Transaction):

    # Convert input to DataFrame
    df = pd.DataFrame([txn.model_dump()])

    # Predict
    prediction = int(model.predict(df)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df)[0][1])

    return {
        "fraud_prediction": prediction,
        "fraud_probability": probability
    }



    

