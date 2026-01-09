import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create dummy training data
X = pd.DataFrame({
    "amount": np.random.rand(100) * 1000,
    "oldbalanceOrg": np.random.rand(100) * 5000,
    "newbalanceOrig": np.random.rand(100) * 4000,
})

y = np.random.randint(0, 2, size=100)

# Build pipeline
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ]
)

# Train
model.fit(X, y)

# Save model
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(model, MODEL_DIR / "fraud_pipeline.joblib")
print("✅ Dummy model saved successfully")
