#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:44:12 2024

@author: divyanshuahuja
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load your data
file_path = '/Users/divyanshuahuja/Desktop/DataMining class Bhavesh Shah/NBA-Dfile.xlsx'
data = pd.read_excel(file_path)

# Clean the salary column and convert it to numeric
data['salary'] = pd.to_numeric(data['salary'].replace('[\$,]', '', regex=True), errors='coerce')

# Group by 'Year' and calculate the mean salary
average_salary_by_year = data.groupby('Year')['salary'].mean()

# Plot the average salary by year
plt.figure(figsize=(12, 6))
average_salary_by_year.plot(kind='line', marker='o')
plt.title('Average Salary Changes from 2000 to 2020')
plt.xlabel('Year')
plt.ylabel('Average Salary')
plt.grid(True)
plt.show()
