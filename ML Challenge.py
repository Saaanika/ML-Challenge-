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
