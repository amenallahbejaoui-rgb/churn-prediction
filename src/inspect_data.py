import pandas as pd

DATA_PATH = "data/churn.csv"

df = pd.read_csv(DATA_PATH)

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATES =====")
print(df.duplicated().sum())

print("\n===== TARGET DISTRIBUTION =====")
print(df["Churn"].value_counts())

print("\n===== TARGET PERCENTAGE =====")
print(df["Churn"].value_counts(normalize=True) * 100)

print("\n===== NUMERICAL SUMMARY =====")
print(df.describe())