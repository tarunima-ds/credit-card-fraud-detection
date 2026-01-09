# src/predict.py

import joblib
import pandas as pd

# Load saved pipeline
model = joblib.load("models/fraud_pipeline.joblib")

# Sample new transaction
new_data = pd.DataFrame([{
    "amount": 90000,
    "transaction_type": "P2P",
    "balance_before": 100000,
    "balance_after": 10000
}])

# Predict
prediction = model.predict(new_data)[0]
probability = model.predict_proba(new_data)[0][1]

print("Fraud Prediction:", prediction)
print("Fraud Probability:", probability)
