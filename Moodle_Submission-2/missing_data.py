import pandas as pd
import numpy as np

# Creating a Sample DataFrame
data = np.arange(40, 55).reshape(5, 3)

df = pd.DataFrame(
    data,
    index=['Laptop', 'Tablet', 'Phone', 'Monitor', 'Printer'],
    columns=['ShopA', 'ShopB', 'ShopC']
)

print(df)
print("----------\n")

# Adding a new column with missing values
df['ShopD'] = np.nan

# Adding a new row with values
df.loc['Camera'] = [60, 61, 62, 63]

# Adding a row containing only NaN values
df.loc['Speaker'] = np.nan

# Adding another empty column
df['ShopE'] = np.nan

# Filling one value in ShopD
df.loc['Laptop', 'ShopD'] = 45

print("The dataframe is:")
print(df)
print("----------\n")

print("The Missing values are:")
print(df.isnull())
print("----------\n")

print("The non-missing values are:")
print(df.notnull())
print("----------\n")

print("The count of missing values in each column:")
print(df.isnull().sum())
print("----------\n")

print("The count of total missing values in each column:")
print(df.isnull().sum().sum())
print("----------\n")

print("The count of available values in each column:")
print(df.count())
print("----------\n")

print("Display of non-Null values of ShopD:")
print(df['ShopD'][df['ShopD'].notnull()])
print("----------\n")

print("Removing Missing values from ShopD:")
print(df['ShopD'].dropna())
print("----------\n")

print("Removing any rows containing NaN:")
print(df.dropna())
print("----------\n")

print("Removing rows where all values are NaN:")
print(df.dropna(how='all'))
print("----------\n")

print("Removing columns where all values are NaN:")
print(df.dropna(how='all', axis=1))
print("----------\n")

print("Dropping columns having fewer than 5 non-missing values:")
print(df.dropna(thresh=5, axis=1))
print("----------\n")

arr = np.array([120, 250, np.nan, 400])

series = pd.Series(arr)

print("NumPy Mean:", arr.mean())
print("----------\n")  

print("Pandas Mean:", series.mean())
print("----------\n")

shopD = df['ShopD']

print("ShopD Sum:")
print(shopD.sum())
print("----------\n")

print("ShopD Mean:")
print(shopD.mean())
print("----------\n")

print("ShopD Cumulative Sum:")
print(shopD.cumsum())
print("----------\n")

filled_df = df.fillna(0)

print("Filling the missing values with 0:")
print(filled_df)
print("----------\n")

print("Original Means:")
print(df.mean(numeric_only=True))
print("----------\n")

print("Means After Filling:")
print(filled_df.mean(numeric_only=True))
print("----------\n")

print("Forward Fill:")
print(df['ShopD'].ffill())
print("----------\n")

print("Backward Fill:")
print(df['ShopD'].bfill())
print("----------\n")