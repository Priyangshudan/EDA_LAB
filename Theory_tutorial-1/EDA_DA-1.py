#Q-1) Task-1
import pandas as pd
df=pd.read_csv("employees.csv")
print("First 5 records:")
print(df.head(5))
print("\n")
print("Last 5 records:")
print(df.tail(5))
print("\n")
print("Dataset Shape:",df.shape)
print("\n")
print("Number of records:",len(df))
print("\n")
print("Number of features in dataset:",len(df.columns))
print("\n")

#Q-2) Task-2
print("Columns Names:")
print(list(df.columns))
print("\n")
print("Data Types:")
print(df.dtypes)
print("\n")
print(df.describe())
print("\n")
print("Average Salary:",df['Salary'].mean())
print("\n")
numeric_std = df.select_dtypes(include=['number']).std()
highest_std_feature = numeric_std.idxmax()
highest_std_value = numeric_std.max()
print("The feature with highest standard deviation in given dataset is",highest_std_feature,"with",highest_std_value)
print("\n")

#Q-3) Task-3
for col in df.columns:
    missing_count = df[col].isnull().sum()
    print(f"Column '{col}': {missing_count} missing values")
