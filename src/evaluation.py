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
df = pd.read_csv('telco_cleaned_validated.csv') # Data source: https://github.com/IBM/telco-customer-churn-on-icp4d


#### Overview of Features
features = ['tenure', 'MonthlyCharges', 'TotalCharges']

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, var in enumerate(features):
    # Plain histogram without the 'hue' parameter
    sns.histplot(data=df, x=var, bins=30, color='skyblue', edgecolor='black', ax=axes[i])

    axes[i].set_title(f'Overall Distribution of {var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Customer Count (Frequency)')

plt.tight_layout()
plt.show()

### Objective 1: What is the descriptive relationship of tenure, monthly charges, and total charges with customer churn status?

plt.figure(figsize=(7, 4))

# Create a temporary copy to fix the numerical labels to "Yes" and "No"
df_plot = df.copy()
df_plot['Churn'] = df_plot['Churn'].map({0: 'No', 1: 'Yes'})

# Store the plot in 'ax'
ax = sns.countplot(data=df_plot, y='Churn', hue='Churn', palette='Set2', legend=False)

# Calculate the total number of rows
total_customers = len(df_plot)

# Loop through the horizontal bars and label them with percentages
for container in ax.containers:
    # Calculate percentages on the fly using a lambda function
    labels = [f'{(bar.get_width() / total_customers * 100):.1f}%' for bar in container]
    ax.bar_label(container, labels=labels, padding=5)

plt.title('Frequency Distribution of Customer Churn')
plt.xlabel('Number of Customers')
plt.ylabel('Churn Status')
plt.xlim(0, df_plot['Churn'].value_counts().max() * 1.15) # Adds breathing room for labels
plt.show()

#### Quick Overview of Feature Means

group_means = df.groupby('Churn', observed=False)[features].mean().T
group_means.columns = ['Stayed (No)', 'Churned (Yes)']
print("Group means of features: ", group_means)

# Correlation Between Feature Variables
df[features].corr()

# Customer Count Frequency
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, var in enumerate(features):
    # Changed to histplot for actual frequencies
    # element="step" or multiple="layer" keeps the overlapping clean
    sns.histplot(data=df_plot, x=var, hue='Churn', palette='Set2',
                 multiple='layer', alpha=0.5, bins=30, ax=axes[i])

    axes[i].set_title(f'Frequency Table of {var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Customer Count (Frequency)') # y-axis is now a true count

plt.tight_layout()
plt.show()
### Objective 2: Can customer churn be predicted using tenure, monthly charges, and total charges?

### Train-Test Split
# Split the data into train and test sets with 70% for training data
# To prevent data leakage, separate CSVs are created.
df_train, df_test = train_test_split(df, train_size=0.7, random_state=60)
print(df_train.sample(5))

# Save the train data
# To prevent from creating duplicate index columns, exclude the default indices
# df_train.to_csv("df_train.csv", index=False)

# Save the test data
# df_test.to_csv("df_test.csv", index=False)

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

### k-Nearest Neighbors Model
# Hyperparameter Tuning: Finding the best k

# Choosing k range from 1 to 70
k_value_min = 1
k_value_max = 70

# Create a list of evenly integer k values to test as hyperparameters
k_list = np.linspace(k_value_min, k_value_max, num=70, dtype=int)

# Scaling the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a dictionary to store the k value for model accuracy
knn_dict = {}

# Loop over all k values
for k_value in k_list:
    k_num = int(k_value)
    if k_num > X_train_scaled.shape[0]:
        break
    # Create a KNN Classificaiton model for the current k
    model = KNeighborsClassifier(n_neighbors=k_num)

    # Fit the model on the train data
    model.fit(X_train_scaled,y_train)

    # Use the trained model to predict on the test data
    y_pred = model.predict(X_test_scaled)

    model_accuracy = accuracy_score(y_test, y_pred)
    #print(f"For k={k_value}, Model Accuracy is {model_accuracy}")

    # Store the model accuracy of each k value in the dictionary
    knn_dict[k_num] = model_accuracy

# Plot a graph which depicts the relation between the k values and model accuracy
plt.figure(figsize=(8,6))
plt.plot(list(knn_dict.keys()), list(knn_dict.values()),'k.-',alpha=0.5,linewidth=2)

# Set the title and axis labels
plt.xlabel('k',fontsize=20)
plt.ylabel('Model Accuracy',fontsize = 20)
plt.title('Test $model_accuracy$ values for different k values - KNN Classificaiton',fontsize=20)
plt.tight_layout()
plt.show()


max_model_accuracy = max(knn_dict.values())
tolerance = 0.005  # Adjust as needed for bias-variance tradeoff, 0.002

# Find smallest k whose accuracy is within the tolerance of the maximum accuracy
best_k_robust = min([k for k, model_accuracy in knn_dict.items() if abs(model_accuracy - max_model_accuracy) <= tolerance])

print(f"Highest Model Accuracy: {max_model_accuracy:.3f}")
print(f"Optimal smallest k achieving higher accuracy with tolerance range of 0.005: {best_k_robust}")

#### KNN Classification Model

# Define your classification model
knn_model = KNeighborsClassifier(n_neighbors=best_k_robust, weights='distance')
# Note: closer neighbors get a much stronger vote than further neighbors, protecting (imbalanced)
# minority clusters from being overwhelmed

# Fit the model on the train data
knn_model.fit(X_train_scaled, y_train)

# Predict and compute the accuracy on the test data
y_pred = knn_model.predict(X_test_scaled)

model_accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy is {model_accuracy}")

### Logistic Regression Model

# Define a Logistic Regression model with max_iter as 10000 and C as 0.1 (leave all other parameters at default values)
log_model = LogisticRegression(max_iter = 10000, C = 0.1, class_weight='balanced') # To balance the classes we use class_weight
# C is the regularization strength; 0.01 is very high. default is 1, 10 to 100 is low regularization

# Fit the Logistic Regression model on the train data
log_model.fit(X_train_scaled, y_train)

## Model Evaluation
### Objective 3: Which classification method, Logistic Regression or k-Nearest Neighbors, provides better predictive performance for customer churn?

# To use the metrics, create a dictionary called metric_scores which has keys 'Accuracy', 'Recall', 'Specificity', 'Precision', and 'F1-score'; the first element being KNN Classification Model corresponding metric score and the second element being Logistic Regression Model corresponding metric score.

metric_scores = {}

#Generate predictions on the scaled test features
y_pred_knn = knn_model.predict(X_test_scaled)
y_pred_log = log_model.predict(X_test_scaled)

# Extract confusion_matrix for kNN
tn_k, fp_k, fn_k, tp_k = confusion_matrix(y_test, y_pred_knn).ravel()

# Extract confusion_matrix for logistic regression
tn_l, fp_l, fn_l, tp_l = confusion_matrix(y_test, y_pred_log).ravel()

metric_scores['Accuracy'] = [
    (tp_k + tn_k) / (tp_k + tn_k + fp_k + fn_k),
    (tp_l + tn_l) / (tp_l + tn_l + fp_l + fn_l)
]

metric_scores['Recall'] = [
    tp_k / (tp_k + fn_k),
    tp_l / (tp_l + fn_l)
]

metric_scores['Specificity'] = [
    tn_k / (tn_k + fp_k),
    tn_l / (tn_l + fp_l)
]

metric_scores['Precision'] = [
    tp_k / (tp_k + fp_k),
    tp_l / (tp_l + fp_l)
]

metric_scores['F1-score'] = [
    (2 * tp_k) / (2 * tp_k + fp_k + fn_k),
    (2 * tp_l) / (2 * tp_l + fp_l + fn_l)
]

# Print to verify the exact structural lengths and keys
print(metric_scores)

# Display your results
df_model_results = pd.DataFrame(metric_scores, index=['kNN Model', 'Logistic Regression Model'])
df_formatted = (df_model_results.T*100).map(lambda x: f"{x:.1f}%")
print("Classification Performance Metrics", df_formatted)

### Model Diagnostics: ROC Curve & AUC Performance Analysis
# Define a kNN classification model with best k
knn = KNeighborsClassifier(n_neighbors=best_k_robust, weights='distance')

# Fit the above model on the train data
knn.fit(X_train_scaled, y_train)

# Predict probabilities for the positive class on the test data using the kNN model
y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]

# Define a Logistic Regression model with max_iter as 10000, C as 0.1, and a random_state of 42
logreg = LogisticRegression(max_iter=10000, C=0.1, random_state=42, class_weight='balanced')


# Fit the Logistic Regression model on the train data
logreg.fit(X_train_scaled, y_train)

# Predict probabilities for the positive class on the test data using the logistic regression model
y_prob_logreg = logreg.predict_proba(X_test_scaled)[:, 1]

#### ROC (Receiver Operating Characteristic) Curve
# Thresholds
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

# Note: This can also be done via sklearn.metrics.roc_curve
knn_thresholds = get_thresholds(y_prob_knn)
logreg_thresholds = get_thresholds(y_prob_logreg)

# False Positive Rate & True Positive Rate
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

# FPR for the kNN at each of its thresholds
knn_fpr = get_fpr(y_test, y_prob_knn, knn_thresholds)
# TPR for the kNN at each of its thresholds
knn_tpr = get_tpr(y_test, y_prob_knn, knn_thresholds)

# TPR for the logistic model at each of its thresholds
logreg_tpr = get_tpr(y_test, y_prob_logreg, logreg_thresholds)
# FPR for the logistic model at each of its thresholds
logreg_fpr = get_fpr(y_test, y_prob_logreg, logreg_thresholds)

# Area Under the Curve (AUC)
# Compute the ROC AUC score of the kNN model
knn_auc = roc_auc_score(y_test, y_prob_knn)

# Compute the ROC AUC score of the Logistic model
logreg_auc = roc_auc_score(y_test, y_prob_logreg)



# Initialize the plot canvas size
fig, ax = plt.subplots(figsize=(12, 8))

# 1. Plot KNN Classifier ROC Curve (Fixed comment: Classification, not Regression)
ax.plot(knn_fpr,
        knn_tpr,
        label=f'KNN Classifier (AUC = {knn_auc:.3f})',
        color='g',
        lw=3)

# 2. Plot Balanced Logistic Regression ROC Curve
ax.plot(logreg_fpr,
        logreg_tpr,
        label=f'Logistic Regression (AUC = {logreg_auc:.3f})',
        color='purple',
        lw=3)

# 3. Threshold annotations configuration
label_kwargs = {}
label_kwargs['bbox'] = dict(
    boxstyle='round, pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.8
)
eps = 0.025 # Visual spacing offset so text labels don't clip the curve line

# Annotate Logistic Regression thresholds
label_kwargs = {}
label_kwargs['bbox'] = dict(
    boxstyle='round, pad=0.2', facecolor='white', edgecolor='lightgray', alpha=0.9
)
eps = 0.03 # Spacing offset

# Fix: Changed step from 15 to 200 to spread out the purple labels
for i in range(0, len(logreg_fpr), 200):
    if i < len(logreg_thresholds):
        threshold = f"{logreg_thresholds[i]:.2f}"
        ax.annotate(threshold, (logreg_fpr[i], logreg_tpr[i] - eps),
                    fontsize=9, color='purple', weight='bold', **label_kwargs)

# Fix: Changed step from 1 to 150 to spread out the green labels
for i in range(0, len(knn_fpr), 150):
    if i < len(knn_thresholds):
        threshold = f"{knn_thresholds[i]:.2f}"
        ax.annotate(threshold, (knn_fpr[i], knn_tpr[i] + eps),
                    fontsize=9, color='green', weight='bold', **label_kwargs)


# 4. Plot diagonal line representing a random classifier
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess Baseline (AUC = 0.500)')

# 5. Graph boundaries, styling, and labels configuration
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=16, labelpad=10)
ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=16, labelpad=10)

# Aligns perfectly with your centered Jupyter Notebook rules
ax.set_title('ROC Curve & Multi-Model Threshold Diagnostics', fontsize=18, fontweight='bold', pad=15)
ax.legend(loc="lower right", fontsize=13, frameon=True, facecolor='white', edgecolor='lightgray')
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('roc_curve_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
