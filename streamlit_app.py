import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# Load model
model = joblib.load("fraud_pipeline.joblib")

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details and get fraud prediction.")

# ---- Inputs (change these to match your dataset columns) ----
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=5000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=4000.0)

# Convert transaction_type to numeric/encoding if your model expects it.
# If your training pipeline already had OneHotEncoder for type, keep it as string.
# If you manually encoded type to numbers during training, then do mapping here.



# Create input dataframe (THIS IS input_df)
# Build input exactly like training data
input_data = {
    "step": 1,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "isFlaggedFraud": 0,
    "CASH_OUT": 0,
    "DEBIT": 0,
    "PAYMENT": 0,
    "TRANSFER": 0
}

# One-hot encode transaction type
input_data[transaction_type] = 1

input_df = pd.DataFrame([input_data])



st.write("### Input Data")
st.dataframe(input_df)

if st.button("Predict Fraud"):
    pred = model.predict(input_df)[0]

    # Probability (if available)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0][1]
        st.write(f"Fraud Probability: **{proba:.4f}**")

    if int(pred) == 1:
        st.error("🚨 Fraud Detected")
    else:
        st.success("✅ Not Fraud")

