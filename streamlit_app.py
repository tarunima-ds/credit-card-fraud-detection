import os
import io
import joblib
import requests
import pandas as pd
import streamlit as st


# Page Config (MUST be first Streamlit command)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="centered"
)

st.write("🚀 Streamlit app started")


# Load Model (Local first, Remote fallback)

@st.cache_resource
def load_model():
    local_model_path = "models/fraud_pipeline.joblib"

    # 1️⃣ Load local model (for local development)
    if os.path.exists(local_model_path):
        st.success("✅ Loaded local model")
        return joblib.load(local_model_path)

    # 2️⃣ Load remote model (for Streamlit Cloud)
    url = "https://github.com/tarunima-ds/credit-card-fraud-detection/releases/download/v1.0/fraud_pipeline.joblib"
    st.warning("⬇️ Downloading model from GitHub Release...")

    response = requests.get(url)
    response.raise_for_status()

    return joblib.load(io.BytesIO(response.content))



# Load model safely

try:
    model = load_model()
except Exception as e:
    st.error("❌ Failed to load model")
    st.exception(e)
    st.stop()

# UI

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict whether a transaction is fraudulent.")


# Inputs (match training columns)

amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=5000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=4000.0)

transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
)

# Build input exactly like training data

input_data = {
    "step": 1,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "isFlaggedFraud": 0,
    "PAYMENT": 0,
    "TRANSFER": 0,
    "CASH_OUT": 0,
    "DEBIT": 0,
    "CASH_IN": 0
}

# One-hot encode transaction type
input_data[transaction_type] = 1

input_df = pd.DataFrame([input_data])

st.subheader("🔎 Input Data")
st.dataframe(input_df)

# Prediction

if st.button("🚨 Predict Fraud"):
    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0][1]
        st.write(f"Fraud Probability: **{probability:.4f}**")

    if int(prediction) == 1:
        st.error("⚠️ Fraud Detected!")
    else:
        st.success("✅ Transaction is NOT Fraudulent")
