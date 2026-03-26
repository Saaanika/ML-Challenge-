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
