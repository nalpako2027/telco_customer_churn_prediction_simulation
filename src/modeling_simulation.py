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


# Read the datafile "df_train.csv"
df_train = pd.read_csv('df_train.csv') 

# Take a quick look at the dataframe
print(df_train.head())

# Read the datafile "fd_test.csv"
df_test = pd.read_csv('df_test.csv')
# Take a quick look at the dataframe
print(df_test.head())


# Get the train predictors
X_train = df_train[['tenure', 'MonthlyCharges', 'TotalCharges']]

# Get the train response variable
y_train = df_train['Churn']


# Get the test predictors
X_test = df_test[['tenure', 'MonthlyCharges', 'TotalCharges']]

# Get the test response variable
y_test = df_test['Churn']

# Scaling the features

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Model Evaluation

# Define a Logistic Regression model with max_iter as 10000, C as 0.1, and a random_state of 42
logreg = LogisticRegression(max_iter=10000, C=0.1, random_state=42, class_weight='balanced') 

# Fit the Logistic Regression model on the train data
logreg.fit(X_train_scaled, y_train)

# Predict probabilities for the positive class on the test data using the logistic regression model
y_prob_logreg = logreg.predict_proba(X_test_scaled)[:, 1]

def get_thresholds(y_pred_proba):
    # Only consider unique predicted probabilities
    unique_probas = np.unique(y_pred_proba)
    # Sort unique probabilities in descending order
    unique_probas_sorted = unique_probas[::-1]

    # To ensure ROC curves reach the corners of the plot, (0,0) and (1,1) add some additional thresholds to the set
    # Insert 1.1 at the beginning of the threshold array: a value greater than 1
    # is required for the ROC curve to reach the lower left corner
    # (0 fpr, 0 tpr) considering one of our models produces probability predictions of 1
    thresholds = np.insert(unique_probas_sorted, 0, 1.1)
    # Append 0 to the end of the thresholds
    thresholds = np.append(thresholds, 0)
    return thresholds

logreg_thresholds = get_thresholds(y_prob_logreg)

# #### FPR & TPR

def get_fpr(y_true, y_pred_proba, threshold):
    # Ensure threshold behaves as a list or array
    if isinstance(threshold, (int, float, np.number)):
        threshold = [threshold]

    # Pre-calculate the total number of actual negatives (denominator) once
    # This completely eliminates the need to run confusion_matrix() over and over
    total_negatives = (y_true == 0).sum()

    fpr_list = []
    for t in threshold: 
        # NumPy vectorized step: count where model predicts 1 but actual is 0
        fp = ((y_pred_proba >= t) & (y_true == 0)).sum()
        fpr_list.append(fp / total_negatives)

    return fpr_list[0] if len(fpr_list) == 1 else np.array(fpr_list)


def get_tpr(y_true, y_pred_proba, threshold):
    if isinstance(threshold, (int, float, np.number)):
        threshold = [threshold]

    total_positives = (y_true == 1).sum()

    tpr_list = []
    for t in threshold: 
        tp = ((y_pred_proba >= t) & (y_true == 1)).sum()
        tpr_list.append(tp / total_positives)

    return tpr_list[0] if len(tpr_list) == 1 else np.array(tpr_list)


# TPR for the logistic model at each of its thresholds
logreg_tpr = get_tpr(y_test, y_prob_logreg, logreg_thresholds)
# FPR for the logistic model at each of its thresholds
logreg_fpr = get_fpr(y_test, y_prob_logreg, logreg_thresholds)


# ## Objective 4: How can predicted churn probabilities be used to identify high-risk customers for targeted retention efforts?

# ### Simulate the Risk Tranches

# Create an intervention simulation DataFrame
sim_df = pd.DataFrame({
    'tenure': X_test['tenure'],
    'MonthlyCharges': X_test['MonthlyCharges'],
    'TotalCharges': X_test['TotalCharges'],
    'Churn_Probability': y_prob_logreg,
    'Actual_Churn': y_test
})

# Define business risk tranches based on probability thresholds from your ROC curve
conditions = [
    (sim_df['Churn_Probability'] >= 0.75),
    (sim_df['Churn_Probability'] >= 0.50) & (sim_df['Churn_Probability'] < 0.75),
    (sim_df['Churn_Probability'] >= 0.25) & (sim_df['Churn_Probability'] < 0.50),
    (sim_df['Churn_Probability'] < 0.25)
]
choices = ['1. Critical Risk', '2. High Risk', '3. Medium Risk', '4. Low Risk']
sim_df['Risk_Segment'] = np.select(conditions, choices, default='4. Low Risk')

# Display the volume and financial exposure of each segment
summary = sim_df.groupby('Risk_Segment').agg(
    Customer_Count=('Churn_Probability', 'count'),
    Avg_Monthly_Bill=('MonthlyCharges', 'mean'),
    Total_Monthly_Revenue_At_Risk=('MonthlyCharges', 'sum')
).round(2)

print(summary)


# ## Objective 5: What is the estimated financial impact of targeting high-risk customers with a retention intervention?

# #### Simulation: Estimating Financial Impact 

# Baseline parameters from your specific test outcomes
total_targeted_customers = 279 + 572  # Critical Risk + High Risk count = 851
total_monthly_revenue_at_risk = 23957.45 + 41003.35  # $64,960.80

# From your results matrix: Precision is 50.76%
# This means out of 851 targeted, 50.76% are actual churners, 49.24% are false alarms
precision_rate = 0.5076  
actual_churners_targeted = int(total_targeted_customers * precision_rate)  # 431 customers
false_alarms_targeted = total_targeted_customers - actual_churners_targeted  # 420 customers

# Strategic Intervention Assumptions (Adjust these variables as needed)
intervention_success_rate = 0.40  # Assume we successfully save 40% of the real churners we target
monthly_retention_offer_cost = 7.00  # We offer a $15/month price drop/incentive to all 851 targeted accounts
customer_lifespan_months = 12  # We measure the financial impact over a 1-year contract extension

# Financial Calculations
customers_saved = int(actual_churners_targeted * intervention_success_rate)  # ~172 customers saved

# Revenue Saved: Saved customers * their average bill ($76.33 weighted avg) * 12 months
avg_bill_targeted = total_monthly_revenue_at_risk / total_targeted_customers
gross_revenue_saved = customers_saved * avg_bill_targeted * customer_lifespan_months

# Campaign Cost: Offering the discount to ALL 851 targeted customers for 12 months
total_campaign_cost = total_targeted_customers * monthly_retention_offer_cost * customer_lifespan_months

# Net Financial Impact
net_revenue_saved = gross_revenue_saved - total_campaign_cost
roi_percentage = (net_revenue_saved / total_campaign_cost) * 100

# --- DISPLAY THE RESULTS ---
print(f"============================================================")
print(f"      FINANCIAL ROI SIMULATION: RETENTION INTERVENTION      ")
print(f"============================================================")
print(f"Total High-Risk Customers Targeted : {total_targeted_customers} accounts")
print(f"True Churners Correctly Flagged    : {actual_churners_targeted} accounts")
print(f"Insurance Operating Cost (Safe Customers)      : {false_alarms_targeted} accounts")
print(f"------------------------------------------------------------")
print(f"Estimated Customers Saved (40% SR) : {customers_saved} accounts")
print(f"Gross Lifetime Revenue Protected   : USD {gross_revenue_saved:,.2f}")
print(f"Total 12-Month Campaign Cost       : USD {total_campaign_cost:,.2f}")
print(f"============================================================")
print(f"NET FINANCIAL BENEFIT (PROFIT)     : USD {net_revenue_saved:,.2f}")
print(f"RETURN ON INVESTMENT (ROI)         : {roi_percentage:.1f}%")
print(f"============================================================")




