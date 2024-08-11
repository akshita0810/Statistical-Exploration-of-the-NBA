import pandas as pd

# Load datasets
df1 = pd.read_excel("NBA_2000_2024.xlsx")
df2 = pd.read_csv("nba-salaries.csv")

# Display null values for df1
print("Null values in df1:")
print(df1.isnull().sum())

# Display null values for df2
print("\nNull values in df2:")
print(df2.isnull().sum())

# Display info() for df1
print("\nDataFrame 1 Info:")
df1.info()

# Display info() for df2
print("\nDataFrame 2 Info:")
df2.info()

# Display data types for df1
print("\nData Types for df1:")
print(df1.dtypes)

# Display data types for df2
print("\nData Types for df2:")
print(df2.dtypes)

# Merge the datasets based on 'Year' and 'Player' in df1 and 'season' and 'name' in df2
df_merged = pd.merge(df1, df2, left_on=['Year', 'Player'], right_on=['season', 'name'], how='left')

# Create df3 with only the columns from df1 and 'salary' from df2
df3 = df_merged[df1.columns.tolist() + ['salary']]

# Create df4 for the unmatched rows
df4 = df_merged[df_merged['salary'].isnull()].drop(columns=['salary'])

# Display the number of records successfully merged
print(f"Number of records successfully merged: {df3.shape[0]}")

# Display info and sample data for df3
print("\ndf3 Info:")
df3.info()
print("\ndf3 Sample Data:")
print(df3.head())

# Display info and sample data for df4
print("\ndf4 Info:")
df4.info()
print("\ndf4 Sample Data:")
print(df4.head())

# Convert df3 to CSV
df3.to_csv("df3.csv", index=False)

print("df3 has been saved as df3.csv")
