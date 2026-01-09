RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL = "isFraud"

DROP_COLS = ["nameOrig", "nameDest"]  # IDs (high cardinality)
CAT_COLS = ["type"]
NUM_COLS = [
    "step", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "isFlaggedFraud"
]

MODEL_PATH = "models/fraud_pipeline.joblib"
DATA_PATH = "data/raw.csv"
