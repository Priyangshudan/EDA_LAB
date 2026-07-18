#Q-1) Task-1
import pandas as pd
df=pd.read_csv("employees.csv")
print("TASK-1")
print("***********\n")
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
print("TASK-2")
print("***********\n")
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
print("TASK-3")
print("***********\n")
for col in df.columns:
    missing_count = df[col].isnull().sum()
    print("Column",col,":",missing_count,"missing values")
print("\n")
for col in df.columns:
    missing_perc = (df[col].isnull().sum() / len(df)) * 100
    print("Column",col,"missing percentage:",missing_perc,"% missing values")
print("\n")
max_missing_count = -1
column_with_most_missing = None
for col in df.columns:
    missing_count = df[col].isnull().sum()
    
    if missing_count > max_missing_count:
        max_missing_count = missing_count
        column_with_most_missing = col

print("The column with the highest number of missing entries is",column_with_most_missing ,"with",max_missing_count, "missing values.")
print("\n")
print("Number of Duplicate Records:",df.duplicated().sum())
print("Dataset Size before Cleaning:",df.shape[0])
df_cleaned = df.drop_duplicates(keep='first').reset_index(drop=True)
print("Dataset Size after Cleaning:",df_cleaned.shape[0])
print("\n")