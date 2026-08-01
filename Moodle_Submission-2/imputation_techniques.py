import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

df = pd.DataFrame({
    "Experience": [2, 5, np.nan, 8, 4],
    "Bonus": [12000, np.nan, 18000, 25000, 15000],
    "Department": ["HR", "Finance", np.nan, "IT", "Finance"]
})

print("Original DataFrame")
print(df)

# 1. Mean Imputation 
mean_imputer = SimpleImputer(strategy="mean")

df["Experience_Mean"] = mean_imputer.fit_transform(df[["Experience"]])

# 2. Median Imputation 
median_imputer = SimpleImputer(strategy="median")

df["Bonus_Median"] = median_imputer.fit_transform(df[["Bonus"]])

# 3. Mode Imputation 
mode_imputer = SimpleImputer(strategy="most_frequent")

df["Department_Mode"] = mode_imputer.fit_transform(df[["Department"]]).ravel()

# 4. MICE Imputation
numeric_cols = ["Experience", "Bonus"]

mice_imputer = IterativeImputer(max_iter=10, random_state=42)

df_mice = df[numeric_cols].copy()

df_mice[:] = mice_imputer.fit_transform(df_mice)

df["Experience_MICE"] = df_mice["Experience"]
df["Bonus_MICE"] = df_mice["Bonus"]

# Final DataFrame
print("\nDataFrame After Imputation")
print(df)