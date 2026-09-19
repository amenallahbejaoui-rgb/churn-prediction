import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


# ==========================================
# 1. Load cleaned data
# ==========================================

df = pd.read_csv("data/churn_clean.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]


# ==========================================
# 2. Features
# ==========================================

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# ==========================================
# 3. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ==========================================
# 4. Final model
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
)


# ==========================================
# 5. Train on ALL available data
# ==========================================

model.fit(X, y)


# ==========================================
# 6. Save model + threshold
# ==========================================

artifact = {
    "model": model,
    "threshold": 0.30
}

joblib.dump(
    artifact,
    "models/churn_model.joblib"
)


print("Model saved successfully.")
print("Path: models/churn_model.joblib")
print("Threshold: 0.30")