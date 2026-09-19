import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. Load data
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
# 3. Train / test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ==========================================
# 5. Model
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
# 6. Train
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 7. Predict probabilities
# ==========================================

probabilities = model.predict_proba(X_test)[:, 1]


# ==========================================
# 8. Apply selected threshold
# ==========================================

THRESHOLD = 0.30

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ==========================================
# 9. Metrics
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

cm = confusion_matrix(
    y_test,
    predictions
)


# ==========================================
# 10. Results
# ==========================================

print("\n==========================================")
print("FINAL TEST RESULTS")
print("==========================================")

print(f"Threshold : {THRESHOLD:.2f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1        : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"]
    )
)