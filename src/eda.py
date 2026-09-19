import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/churn_clean.csv"

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("===== DATASET =====")
print("Shape:", df.shape)

print("\n===== TARGET =====")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True) * 100)

# --------------------------------------------------
# Numerical features
# --------------------------------------------------

numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

print("\n===== NUMERICAL FEATURES =====")
print(df[numerical_features].describe())

# --------------------------------------------------
# Churn rate by categorical variables
# --------------------------------------------------

categorical_features = [
    "gender",
    "SeniorCitizen",
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

for feature in categorical_features:

    print(f"\n===== {feature} =====")

    result = (
        df.groupby(feature)["Churn"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )

    result["churn_rate_%"] = result["mean"] * 100

    print(result)

# --------------------------------------------------
# Visualization 1: Churn distribution
# --------------------------------------------------

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Visualization 2: Churn vs tenure
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure vs Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (months)")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Visualization 3: Churn vs MonthlyCharges
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Visualization 4: Contract vs Churn
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="Contract",
    y="Churn"
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract")
plt.ylabel("Churn Rate")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Visualization 5: Internet Service vs Churn
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="InternetService",
    y="Churn"
)

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Correlation matrix
# --------------------------------------------------

numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Numerical Feature Correlation")

plt.tight_layout()
plt.show()