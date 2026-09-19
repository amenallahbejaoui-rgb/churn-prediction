import pandas as pd

INPUT_PATH = "data/churn.csv"
OUTPUT_PATH = "data/churn_clean.csv"

# Load dataset
df = pd.read_csv(INPUT_PATH)

print("Original shape:", df.shape)

# --------------------------------------------------
# 1. Convert TotalCharges to numeric
# Invalid/blank values become NaN
# --------------------------------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing TotalCharges after conversion:")
print(df["TotalCharges"].isna().sum())

# --------------------------------------------------
# 2. Investigate rows with missing TotalCharges
# --------------------------------------------------

print("\nRows with missing TotalCharges:")
print(
    df[df["TotalCharges"].isna()][
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)

# --------------------------------------------------
# 3. Remove rows where TotalCharges is missing
# These correspond to customers with no accumulated charges
# --------------------------------------------------

df = df.dropna(subset=["TotalCharges"])

# --------------------------------------------------
# 4. Remove customerID
# It identifies the customer but has no predictive meaning
# --------------------------------------------------

df = df.drop(columns=["customerID"])

# --------------------------------------------------
# 5. Convert target to binary
# No = 0
# Yes = 1
# --------------------------------------------------

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# --------------------------------------------------
# 6. Save cleaned dataset
# --------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\n===== CLEANING RESULT =====")
print("Final shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["Churn"].value_counts())

print("\nData types:")
print(df.dtypes)