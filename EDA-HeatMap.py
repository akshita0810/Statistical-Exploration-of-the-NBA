import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
file_path = '/Users/divyanshuahuja/Desktop/DataMining class Bhavesh Shah/NBA-Dfile.xlsx'
data = pd.read_excel(file_path)

# Assuming 'salary' is the column with salary information and may contain non-numeric characters
# Clean the salary column by removing any currency symbols or other characters and convert it to numeric
data['salary'] = pd.to_numeric(data['salary'].astype(str).str.replace('[\$,]', ''), errors='coerce')

# Drop rows with NaN values that resulted from the conversion
data.dropna(subset=['salary'], inplace=True)

# Calculate FG% if not already present and if 'FG' and 'FGA' columns exist
if 'FG%' not in data.columns and 'FG' in data.columns and 'FGA' in data.columns:
    data['FG%'] = data['FG'] / data['FGA']

# Fill in missing values with column means for numeric columns only
data.fillna(data.select_dtypes(include=[np.number]).mean(), inplace=True)

# Calculate the correlation matrix for numeric columns only
correlation_matrix = data.select_dtypes(include=[np.number]).corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Get the absolute correlation values for the 'salary' column and sort them
salary_correlations = correlation_matrix['salary'].abs().sort_values(ascending=False)

# Print the features with the highest absolute correlation with 'salary'
print("Features most highly correlated with salary:")
print(salary_correlations.head())