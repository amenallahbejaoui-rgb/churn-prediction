import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. Load data
# ============================================================

df = pd.read_csv("data/churn_clean.csv")

X = df.drop(columns=["Churn"])
y = df["Churn"]


# ============================================================
# 2. Features
# ============================================================

numerical_features = [
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


# ============================================================
# 3. Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. Preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 5. Model
# ============================================================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ]
)


# ============================================================
# 6. Train
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 7. Get feature names
# ============================================================

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    model
    .named_steps["classifier"]
    .feature_importances_
)


# ============================================================
# 8. Create importance dataframe
# ============================================================

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)


# ============================================================
# 9. Print top features
# ============================================================

print("\n==============================")
print("TOP 20 FEATURES")
print("==============================")

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 10. Plot
# ============================================================

top_features = importance_df.head(15).sort_values(
    "importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()