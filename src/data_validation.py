#!/usr/bin/env python
# coding: utf-8

# Customer Churn Prediction and Retention Strategy Analysis with Business Impact Simulation

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import seaborn as sns


# Read the Telco datafile
df = pd.read_csv('Telco-Customer-Churn.csv') # Data source: https://github.com/IBM/telco-customer-churn-on-icp4d

# Take a quick look at the dataframe
print(df['Churn'].value_counts())
print(df.head())
print(df.columns)
df = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]

### Diagnostics
def var_diagnostic(df, var):
    """Diagnostic information about an individual variable"""
    print(f"--------------- Diagnostic Info: {var} ---------------" )
    print("Descriptive info: \n", df[var].describe())
    print("Unique Values: \n", df[var].unique()[: 10])
    print("Variable Type: \n", df[var].dtypes)
    print("Missing values: \n", df[var].isna().sum())
    print("------------------------- End -------------------------" )

var_diagnostic(df, 'TotalCharges')
var_diagnostic(df, 'MonthlyCharges')
var_diagnostic(df, 'tenure')
var_diagnostic(df, 'Churn')

# For potential/hidden empty spaces
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

# Convert Churn into categorical and Total Charges (int) into float
df['Churn'] = df['Churn'].astype('category')
df['Churn'] = df['Churn'].map({'Yes':1, 'No': 0})
df['TotalCharges'] = df['TotalCharges'].astype('float')

### Variables
var_diagnostic(df, 'TotalCharges')
var_diagnostic(df, 'MonthlyCharges')
var_diagnostic(df, 'tenure')
var_diagnostic(df, 'Churn')

features = ['tenure', 'MonthlyCharges', 'TotalCharges']
print(df[features].shape)

### Handling Missing Data
df['TotalCharges'] = df['TotalCharges'].fillna(0)  # Fills new customers with 0
df['TotalCharges'].isna().sum()

### Diagnostics for Outliers
# Create a figure with 1 row and 3 subplots side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Create a temporary copy to keep labels as "No" and "Yes"
df_plot = df.copy()
df_plot['Churn'] = df_plot['Churn'].map({0: 'No', 1: 'Yes'})

for i, var in enumerate(features):
    # x is the grouping variable, y is the continuous feature
    # legend=False turns off individual legend boxes since the x-axis is already clear
    sns.boxplot(data=df_plot, x='Churn', y=var, hue='Churn',
                palette='Set2', ax=axes[i], legend=False)

    axes[i].set_title(f'{var} Boxplot by Churn')
    axes[i].set_xlabel('Churn Status')
    axes[i].set_ylabel(var)

plt.tight_layout() # Prevents labels from overlapping between plots
plt.show()

# Isolate the churned customers
churned = df[df['Churn'].isin([1])].copy()
def outliers_detection(var, features):
    """Calculating and reporting outliers"""
    # Calculate Interquartile range (IQR) components for TotalCharges
    q1 = churned[var].quantile(0.25)
    q3 = churned[var].quantile(0.75)
    iqr = q3-q1

    # Define the upper outlier fence
    upper_fence = q3 + (1.5*iqr)
    print(f"\nStatistically, any churned customer with {var} value over ${upper_fence:.2f} is an outlier.\n")

    # Filter using the exact statistical boundary
    var_outlier = churned[churned[var] > upper_fence]

    print(f"Found {len(var_outlier)} statistical outliers for {var}:")
    print(var_outlier[features].head())
    print("-------------------------------------")

outliers_detection('TotalCharges', features)
outliers_detection('MonthlyCharges', features)
outliers_detection('tenure', features)

# **Checking the Validity of Outliers:**

# Isolate churned customers
churned = df[df['Churn'].isin([1])].copy()

# Find the statistical outlier threshold
q1 = churned['TotalCharges'].quantile(0.25)
q3 = churned['TotalCharges'].quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + (1.5 * iqr)

# Filter for just the outliers
outliers = churned[churned['TotalCharges'] > upper_fence].copy()

print(f"Analyzing {len(outliers)} total outliers...")

# LOGIC TEST: Calculate expected charges
# We use .get_width or direct math: tenure * MonthlyCharges
outliers['Expected_Total'] = outliers['tenure'] * outliers['MonthlyCharges']

# Calculate the percentage difference between reality and theory
outliers['Percent_Difference'] = (
    (outliers['TotalCharges'] - outliers['Expected_Total']).abs() / outliers['Expected_Total']) * 100

# 6. Flag rows where the mismatch is greater than 15% (System Errors)
# If tenure is 0, Expected_Total is 0, which can cause a divide-by-zero (handled safely)
system_errors = outliers[outliers['Percent_Difference'] > 15]
valid_outliers = outliers[outliers['Percent_Difference'] <= 15]

print(f"✅ Valid Outliers found: {len(valid_outliers)} (Real high-value customers to KEEP)")
print(f"❌ System Errors found: {len(system_errors)} (Data glitches to REMOVE or FIX)")

if len(system_errors) > 0:
    print("\nReviewing the data glitches:")
    print(system_errors[['tenure', 'MonthlyCharges', 'TotalCharges', 'Expected_Total', 'Percent_Difference']])

### Data Quality Assurance & Logic Assertion Checks
# Negative Balance Check
assert df[(df['MonthlyCharges'] <0) | (df['TotalCharges'] < 0)].shape[0] == 0, \
    "Data Error: Negative charges exist in the dataset!"
assert df[df['tenure'] < 0].shape[0] == 0, \
    "Data Error: Negative tenure values!"

# New Customer Balance Check
assert df[(df['tenure'] ==0) & (df['TotalCharges'] > 0)].shape[0] ==0, \
    "Logic Error: Customers with 0 tenure has Total Charges > $0"

# Total Charges Check related to being equivalent to (Tenure * Monthly Charges)
# A previous check showed that two cases revealed Logic Error, which might be due
# to starting with a more expensive product and then switching it.
# We should filter for tenure > 5
assert len(df[(df['tenure'] > 5) & (df['TotalCharges'] > (df['tenure'] * df['MonthlyCharges'] * 1.5))]) == 0, \
    "Logic Error: TotalCharges is physically impossible relative to tenure and monthly rates!"

df.to_csv("telco_cleaned_validated.csv", index=False)