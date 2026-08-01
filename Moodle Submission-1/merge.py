import pandas as pd
import numpy as np

# Software Engineering Data
df1SE = pd.DataFrame({
    'StudentID': [101, 103, 105, 107, 109, 111, 113, 115],
    'ScoreSE': [72, 65, 81, 54, 90, 76, 69, 88]
})

df2SE = pd.DataFrame({
    'StudentID': [102, 104, 106, 108, 110, 112, 114, 116],
    'ScoreSE': [84, 59, 73, 95, 67, 80, 62, 91]
})

# Machine Learning Data
df1ML = pd.DataFrame({
    'StudentID': [101, 102, 103, 105, 107, 109, 111, 113],
    'ScoreML': [75, 68, 82, 79, 63, 91, 70, 86]
})

df2ML = pd.DataFrame({
    'StudentID': [104, 106, 108, 110, 112],
    'ScoreML': [58, 77, 94, 81, 69]
})

# Combine SE Data
dfSE = pd.concat([df1SE, df2SE], ignore_index=True)

# Combine ML Data
dfML = pd.concat([df1ML, df2ML], ignore_index=True)

#Combine ML SE Data
df = pd.concat([dfML, dfSE], axis=1)
print(df)
print("-----------\n")

df = dfSE.merge(dfML, how='inner', on='StudentID')
print(df)
print("-----------\n")

df = dfSE.merge(dfML, how='left', on='StudentID')
print(df)
print("-----------\n")

df = dfSE.merge(dfML, how='right', on='StudentID')
print(df)
print("-----------\n")


total_students = len(pd.concat([
    dfSE[['StudentID']],
    dfML[['StudentID']]
]).drop_duplicates())

print("Total unqiue students:", total_students)
print("-----------\n")

only_se = dfSE.merge(dfML, on='StudentID', how='left', indicator=True)
only_se = only_se[only_se['_merge'] == 'left_only']

print("Only Software Engineering Students:")
print(only_se[['StudentID', 'ScoreSE']])
print("-----------\n")

only_ml = dfML.merge(dfSE, on='StudentID', how='left', indicator=True)
only_ml = only_ml[only_ml['_merge'] == 'left_only']

print("Only Machine Learning Students:")
print(only_ml[['StudentID', 'ScoreML']])
print("-----------\n")

sales = pd.DataFrame({
    'Company': ['Dell', 'HP', 'Lenovo', 'Apple', 'Dell', 'HP'],
    'Product': ['Laptop', 'Desktop', 'Monitor', 'Tablet', 'Keyboard', 'Mouse'],
    'Quantity': [5, 3, 8, 2, 10, 15],
    'UnitPrice': [60000, 45000, 12000, 35000, 1500, 800]
})

# Add TotalPrice column
sales['TotalPrice'] = sales['Quantity'] * sales['UnitPrice']

print(sales)
print("-----------\n")

print(sales['Company'].value_counts())
print("-----------\n")

print(sales.describe())
print("-----------\n")