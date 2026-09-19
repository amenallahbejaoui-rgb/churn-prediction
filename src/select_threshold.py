import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


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
# 3. Train / final test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Final test samples:", len(X_test))


# ==========================================
# 4. Logistic Regression pipeline
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
# 5. Out-of-fold probabilities
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]


# ==========================================
# 6. Test different thresholds
# ==========================================

thresholds = np.arange(0.20, 0.71, 0.05)

results = []

for threshold in thresholds:

    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_train,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_train,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_train,
        predictions,
        zero_division=0
    )

    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    )


results_df = pd.DataFrame(results)


# ==========================================
# 7. Display results
# ==========================================

print("\n==========================================")
print("Threshold Analysis")
print("==========================================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1": "{:.4f}".format
        }
    )
)


# ==========================================
# 8. Best threshold by F1
# ==========================================

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n==========================================")
print("Best Threshold")
print("==========================================")

print(
    f"Threshold : {best_row['threshold']:.2f}"
)

print(
    f"Precision : {best_row['precision']:.4f}"
)

print(
    f"Recall    : {best_row['recall']:.4f}"
)

print(
    f"F1        : {best_row['f1']:.4f}"
)