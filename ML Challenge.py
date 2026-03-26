# Loading libraries 
import pandas as pd
import numpy as np


# --------------------------------------------- DIN INFORMATION -------------------------------------------------------------------
# Loading datasets

claims = pd.read_csv("claims.csv")
outcomes_train = pd.read_csv("outcomes_train.csv")


print(claims["DIN"].describe())
print(claims["DIN"])


# Checking the distribution of DIN lengths to understand formatting issues
print(claims["DIN"].str.len().value_counts())


# Removing hyphens and spaces from DIN

claims["DIN"] = (
    claims["DIN"]
    .astype(str)
    .str.replace("-", "")
    .str.replace(" ", "")
)


# Checking the distribution of DIN lengths to understand formatting issues
print(claims["DIN"].str.len().value_counts())


# Removing the 3 leading 0s for DIN who have a length of 8 (vs 5 for those without the 0s)
for i, din in enumerate(claims["DIN"]):
    if len(din) == 8:
        claims.at[i, "DIN"] = din[3:]



# --------------------------------------------- Cost Reconciliation -------------------------------------------------------------------

# 1) calculate what total_cost should be
claims["expected_total"] = claims["drug_cost"] + claims["dispense_fee"]

# 2) compare recorded total_cost vs expected_total
claims["cost_diff"] = claims["total_cost"] - claims["expected_total"]

# 3) flag bad records using a small rounding tolerance
tolerance = 0.01
claims["financial_mismatch"] = claims["cost_diff"].abs() > tolerance

# 4) see how many mismatches there are
print("Number of mismatched claims:", claims["financial_mismatch"].sum())

# 5) display the bad rows
bad_claims = claims.loc[
    claims["financial_mismatch"],
    ["claim_id", "drug_cost", "dispense_fee", "total_cost", "expected_total", "cost_diff"]
]

bad_claims.head(10)


# percentage of records with mismatch
pct = claims["financial_mismatch"].mean() * 100
print(f"Percent mismatched: {pct:.2f}%")

# quick stats on the differences
claims.loc[claims["financial_mismatch"], "cost_diff"].describe()




# fix total_cost using components
claims_clean = claims.copy()

claims_clean.loc[
    claims_clean["financial_mismatch"],
    "total_cost"
] = claims_clean["expected_total"]

# --------------------------------------------- DEMOGRAPHIC COLUMNS -------------------------------------------------------------------
# Checking and Fixing Gender
claims_clean['gender'].unique()
claims_clean["gender"] = claims_clean["gender"].replace({
    "M": "male",
    "Male": "male",
    "F": "female",
    "F ": "female",
    "X": np.nan,
    "OTHER": "other",
    "nan": np.nan,
    "Unknown": np.nan
})

# Checking and Fixing Age
claims_clean["age"] = pd.to_numeric(claims_clean["age"], errors="coerce").astype("Int64")
claims_clean["age"].describe()
claims_clean["age"].unique()
claims_clean.loc[claims_clean["age"] < 1, "age"] = None
claims_clean.loc[claims_clean["age"] > 153, "age"] = None

# Checking and Fixing patient_name
claims_clean['patient_name'].unique()
claims_clean[claims_clean["patient_name"].str.contains(r"[^a-zA-Z\s]", na=False)]
claims_clean["patient_name"] = claims_clean["patient_name"].str.replace(
    r"^\s*([^,]+)\s*,\s*(.+)$",
    r"\2 \1",
    regex=True
)
claims_clean["patient_name"] = claims_clean["patient_name"].str.title()



## --------------------------------------------- Duplicate claims --------------------------------------------------------
# check duplicate claim IDs
claims_clean["duplicate_claim_id"] = claims_clean.duplicated(
    subset=["claim_id"], keep=False
)

print("Duplicate claim IDs:", claims_clean["duplicate_claim_id"].sum())

# remove duplicate claim IDs
claims_clean = claims_clean.drop_duplicates(subset=["claim_id"], keep="first")

# check near-duplicate claim events
key_cols = ["patient_name", "DIN", "dispense_date", "pharmacy_id", "provider_id"]

claims_clean["is_duplicate_event"] = claims_clean.duplicated(
    subset=key_cols, keep=False
)

print("Potential duplicate claim events:", claims_clean["is_duplicate_event"].sum())

# inspect them
claims_clean[claims_clean["is_duplicate_event"]].sort_values(key_cols).head(20)

# remove near-duplicates if appropriate
claims_clean = claims_clean.drop_duplicates(subset=key_cols, keep="first")


## --------------------------------------------- Date Logic --------------------------------------------------------


# Fix MM/DD/YYYY → YYYY-MM-DD using explicit format
mask_slash = claims_clean["dispense_date"].str.match(r"^\d{2}/\d{2}/\d{4}$", na=False)
claims_clean.loc[mask_slash, "dispense_date"] = pd.to_datetime(
    claims_clean.loc[mask_slash, "dispense_date"], format="%m/%d/%Y"
).dt.strftime("%Y-%m-%d")

# Fix Mon DD YYYY → YYYY-MM-DD using explicit format
mask_text = claims_clean["dispense_date"].str.match(r"^[A-Za-z]{3} \d{2} \d{4}$", na=False)
claims_clean.loc[mask_text, "dispense_date"] = pd.to_datetime(
    claims_clean.loc[mask_text, "dispense_date"], format="%b %d %Y"
).dt.strftime("%Y-%m-%d")


# No future dates
max_date = pd.Timestamp.today()

# Parse the now-standardized column to Timestamps for comparison
parsed = pd.to_datetime(claims_clean["dispense_date"], format="%Y-%m-%d", errors="coerce")

# Flag anything outside the range (date in the future)
out_of_range = parsed.isna() | (parsed > max_date)

# Set invalid rows to NA
claims_clean.loc[out_of_range, "dispense_date"] = pd.NA





# ---------------------------------------------
# Merge cleaned claims data with training labels
# ---------------------------------------------
df = claims_clean.merge(outcomes_train, on="claim_id", how="inner")

# 1. Decide your target column
target_col = "high_risk"

# 2. Choose your feature columns
# Start with simple, reasonable predictors
feature_cols = [
    "DIN",
    "gender",
    "age",
    "drug_cost",
    "dispense_fee",
    "total_cost",
    "patient_name",
    "dispense_date",
    "region",
    "quantity", "insurer_id",
    "condition", "provider_id", "certificate_id",
]


# 3. Build X and y
X = df[feature_cols].copy()
y = df[target_col].copy()

# 4. Convert date into useful numeric features
X["dispense_date"] = pd.to_datetime(X["dispense_date"], errors="coerce")
X["dispense_year"] = X["dispense_date"].dt.year
X["dispense_month"] = X["dispense_date"].dt.month
X["dispense_dayofweek"] = X["dispense_date"].dt.dayofweek

# Drop raw date column after extracting parts
X = X.drop(columns=["dispense_date"])

# Optional: fill missing values before encoding
X["age"] = X["age"].fillna(X["age"].median())
X["dispense_year"] = X["dispense_year"].fillna(X["dispense_year"].median())
X["dispense_month"] = X["dispense_month"].fillna(X["dispense_month"].median())
X["dispense_dayofweek"] = X["dispense_dayofweek"].fillna(X["dispense_dayofweek"].median())

# Fill missing categorical values
X["gender"] = X["gender"].fillna("missing")
X["patient_name"] = X["patient_name"].fillna("missing")
X["DIN"] = X["DIN"].fillna("missing")

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Preview
print(X.head())
print(y.head())
print(X.shape)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(random_state=42, n_estimators=300)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("\nClassification Report:\n", classification_report(y_test, rf_pred))

