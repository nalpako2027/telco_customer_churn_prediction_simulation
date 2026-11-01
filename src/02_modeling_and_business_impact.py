#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align: center;"> Customer Churn Prediction and Retention Strategy Analysis with Business Impact Simulation </h1>
# 


# In[1]:


# Imported Libraries
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.contingency_tables import mcnemar

from scipy import stats
from scipy.special import logit
from prettytable import PrettyTable


# ---  
print("#--"*10)
print("\n Objective 3: ")
print("\n Using the top-ranked features, can customer churn be predicted, and which classifier — Logistic Regression or k-Nearest Neighbors — performs better?")

# In[2]:


# Load train-test data from Notebook 1
df_train = pd.read_csv('df_train.csv')
df_test = pd.read_csv('df_test.csv')

with open('selected_features.json') as f:
    selected_features = json.load(f)

print(f"Train shape: {df_train.shape}")
print(f"Test shape:  {df_test.shape}")
print(f"Selected features ({len(selected_features)}):")
for i, feature in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feature}")


# In[3]:

# Separate predictors from target
X_train = df_train[selected_features]
y_train = df_train['Churn']
X_test = df_test[selected_features]
y_test = df_test['Churn']

print(f"X_train: {X_train.shape}   y_train churn rate: {y_train.mean():.4f}")
print(f"X_test:  {X_test.shape}   y_test churn rate:  {y_test.mean():.4f}")


print("#--"*10)
print("\n k-Nearest Neighbors Classification Model:")

print("\n**Hyperparameter Tuning: Finding the best k**")

# In[4]:

# Choosing k range from 1 to 100
k_value_min = 1
k_value_max = 100

# In[5]:

# Scaling the features

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# In[6]:

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"Min per column: {X_train_scaled.min(axis=0)}")
print(f"Max per column: {X_train_scaled.max(axis=0)}")

# In[7]:

knn_pipe = make_pipeline(MinMaxScaler(), KNeighborsClassifier(weights='distance'))
knn_grid = GridSearchCV(
    estimator=knn_pipe,
    param_grid={'kneighborsclassifier__n_neighbors': range(k_value_min, k_value_max + 1)},
    scoring='average_precision',
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1
)
knn_grid.fit(X_train, y_train)   # unscaled but scaling happens inside each fold

results = pd.DataFrame(knn_grid.cv_results_)
mean_scores = results['mean_test_score']
params = results['param_kneighborsclassifier__n_neighbors']

max_score = mean_scores.max()

# Find the best k values within 0.5% tolerance of the peak
tolerance = 0.005 
valid_ks = params[mean_scores >= (max_score - tolerance)]
best_k_robust = valid_ks.min()

print(f"Peak CV PR-AUC:          {max_score:.4f} (at k={params[mean_scores.idxmax()]})")
print(f"Chosen k (robust):       {best_k_robust}")

print("#--"*10)
print("\n Performing KNN Model:")

# In[8]:

# Define the classification model
knn_model = KNeighborsClassifier(n_neighbors=best_k_robust, weights='distance')
knn_model.fit(X_train_scaled, y_train)

# In[9]:

# Predict and compute the accuracy on the test data
y_pred_knn = knn_model.predict(X_test_scaled)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"KNN Model Accuracy: {knn_accuracy:.4f}")


print("#--"*10)
print("\nLogistic Regression Model:")

# ### Model Assumptions for Logistic Regression
# 
# The assumptions of logistic regression (Lomax \& Hahs-Vaughn, 2012) are somewhat relaxed compared to OLS regression, but four primary assumptions must still be considered:
# 
# 1. **Noncollinearity.** Predictors should not be highly correlated with one another, as this inflates standard errors and makes coefficient estimates unstable.
#    - This dataset includes `tenure`, `MonthlyCharges`, and `TotalCharges`, which are structurally related (`TotalCharges ≈ tenure × MonthlyCharges`). *How the model handles this overlap will be evaluated after the analysis and further decisions will be made based on the analysis results.* ⏳
# 2. **Linearity (of the logit).** The log-odds of the outcome must be a linear function of the predictors.
#    - *Whether this holds for all features (especially `tenure`) will be discussed after the model analysis.* ⏳
# 3. **Independence of errors.** Each observation should be independent of the others.
#    - This is satisfied by design: each row represents a distinct customer, with no repeated measures over time or across accounts. ✅
# 4. **Values of $X$ are fixed.** The predictors are assumed to be measured without error, or their errors are independent of the outcome.
#    - This is treated as satisfied for a predictive model: billing amounts and tenure are administrative records, not noisy self-reports. ✅

# In[10]:

X_vif = sm.add_constant(X_train.astype(float))
vif_data = pd.DataFrame({
    "Feature": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
vif_data = (vif_data[vif_data["Feature"] != "const"]
            .sort_values("VIF", ascending=False).reset_index(drop=True))
print(vif_data)

# In[11]:

# Tune Logistic Regression using Cross-Validation on the training set
print("Tuning Logistic Regression...")
lr_pipe = make_pipeline(MinMaxScaler(), LogisticRegression(max_iter=10000, random_state=42))
lr_grid = GridSearchCV(
    estimator=lr_pipe,
    param_grid={'logisticregression__C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}, 
    scoring='average_precision',
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1
)
lr_grid.fit(X_train, y_train)
best_C = lr_grid.best_params_['logisticregression__C']
log_model = LogisticRegression(C=best_C, max_iter=10000, random_state=42).fit(X_train_scaled, y_train)
print(f"Best C found via CV: {best_C}")
print(f"Best CV PR-AUC:      {lr_grid.best_score_:.4f}")


# ---  
print("#--"*10)
print("\nModel Comparison:")

# In[12]:

metric_scores = {}

# 1. Get KNN probabilities
y_prob_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

# 2. Logistic regression probabilities (unweighted model, already on the natural scale)
y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

cv5 = StratifiedKFold(5, shuffle=True, random_state=42)

knn_oof_pipe = make_pipeline(
    MinMaxScaler(),
    KNeighborsClassifier(n_neighbors=best_k_robust, weights='distance')
)
log_oof_pipe = make_pipeline(
    MinMaxScaler(),
    LogisticRegression(C=best_C, max_iter=10000, random_state=42)
)

# raw X_train: the scaler is re-fit on the training part of each fold only
oof_knn = cross_val_predict(knn_oof_pipe, X_train, y_train, cv=cv5, method='predict_proba')[:, 1]
oof_log = cross_val_predict(log_oof_pipe, X_train, y_train, cv=cv5, method='predict_proba')[:, 1]

# Threshold search
grid = np.arange(0.10, 0.71, 0.01)
thr_knn = grid[np.argmax([f1_score(y_train, oof_knn >= t) for t in grid])]
thr_log = grid[np.argmax([f1_score(y_train, oof_log >= t) for t in grid])]
print(f"Decision thresholds on the predicted churn probability, chosen on training CV (max F1): KNN={thr_knn:.2f}, LR={thr_log:.2f}")

y_pred_knn = (y_prob_knn >= thr_knn).astype(int)
y_pred_log = (y_prob_log >= thr_log).astype(int)

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

print("#--"*10)
print("\n Confusion matrices:")

# In[13]:

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, y_pred) in zip(axes, [
    (f'KNN (k={best_k_robust}, distance-weighted)', y_pred_knn),
    (f'Logistic Regression (C={log_model.C})', y_pred_log),
]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', cbar=False,
                xticklabels=['Pred: No', 'Pred: Yes'],
                yticklabels=['True: No', 'True: Yes'],
                annot_kws={'size': 14}, ax=ax)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('images/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# In[14]:

# Sanity check: all metric values are valid probabilities
for metric, scores in metric_scores.items():
    assert all(0 <= s <= 1 for s in scores), f"{metric} out of range"

assert set(metric_scores.keys()) == {'Accuracy', 'Recall', 'Specificity', 'Precision', 'F1-score'}, \
    "Unexpected metric keys"
print("✅ All metrics within [0, 1] and expected keys present.")

# In[15]:

# Display the results
df_model_results = pd.DataFrame(metric_scores, index=['kNN Model', 'Logistic Regression Model'])
df_formatted = (df_model_results.T*100).map(lambda x: f"{x:.1f}%")
print(df_formatted)

# In[16]:

# Diagnostic for the multicollinearity problem: Does Dropping TotalCharges Matter?
cv5 = StratifiedKFold(5, shuffle=True, random_state=42)

# 1) What the fitted (penalized) model actually estimates
print("Fitted (penalized) coefficients: ")

print(pd.Series(log_model.coef_[0], index=selected_features)
        [['tenure', 'TotalCharges', 'MonthlyCharges']].round(2))

# 2) VIF after removing the redundant TotalCharges
print("\nVIFs after removing the redundant TotalCharges: ")

feats_noTC = [f for f in selected_features if f != 'TotalCharges']
Xv = sm.add_constant(X_train[feats_noTC].astype(float))
print(pd.Series([variance_inflation_factor(Xv.values, i) for i in range(1, Xv.shape[1])],
                index=feats_noTC).round(2).sort_values(ascending=False).head(4))

# 3) Predictive cost of dropping TotalCharges
print("\nPredictive cost of dropping TotalCharges:")

for name, cols in [('all 13 features', selected_features), ('without TotalCharges', feats_noTC)]:
    pipe = make_pipeline(MinMaxScaler(),
                         LogisticRegression(C=best_C, max_iter=10000, random_state=42))
    oof = cross_val_predict(pipe, X_train[cols], y_train, cv=cv5, method='predict_proba')[:, 1]
    print(f"{name:22s} PR-AUC={average_precision_score(y_train, oof):.4f}  "
          f"ROC-AUC={roc_auc_score(y_train, oof):.4f}  log-loss={log_loss(y_train, oof):.4f}")

# ---  
print("#--"*10)
print("\n Logistic Regression Coefficients — Feature Interpretation:")

# In[17]:


coef_df = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': log_model.coef_[0] # coefficients in log-odds space (features were MinMax-scaled)
}).sort_values('Coefficient', key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#d62728' if c > 0 else '#2ca02c' for c in coef_df['Coefficient']]
bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Logistic Regression Coefficients (scaled features)', fontsize=13, fontweight='bold')
ax.set_xlabel('Coefficient (impact on log-odds of churn)')

# Add numeric labels to each bar
ax.bar_label(bars, labels=[f'{c:+.2f}' for c in coef_df['Coefficient']], 
             padding=5, fontsize=9)

# Extend x-axis limits 
ax.set_xlim(coef_df['Coefficient'].min() * 1.15, coef_df['Coefficient'].max() * 1.15)

plt.tight_layout()
plt.savefig('images/lr_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()

# In[18]:

# A second logistic regression, using statsmodels instead of scikit-learn; 
# statsmodels gives intervals and p-values, which sklearn doesn’t provide directly
Xc = sm.add_constant(pd.DataFrame(X_train_scaled, columns=selected_features))
logit_fit = sm.Logit(y_train.values, Xc).fit(disp=0)
ci = logit_fit.conf_int()
coef_table = pd.DataFrame({'coef': logit_fit.params, 'ci_low': ci[0],
                           'ci_high': ci[1], 'p_value': logit_fit.pvalues}).round(3)
print(coef_table)


# In[19]:

print("#--"*10)
print("\n Linearity assumption check:")
d = X_train.astype(float).copy()
d.columns = d.columns.str.replace(r'\W+', '_', regex=True)
d['Churn'] = y_train.values
terms = list(d.columns.drop('Churn'))
base = smf.logit('Churn ~ ' + ' + '.join(terms), d).fit(disp=0)

for v in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    f = 'Churn ~ ' + ' + '.join([f'cr({t}, df=4)' if t == v else t for t in terms]) # cr a natural cubic spline with 4 degrees of freedom
    sp = smf.logit(f, d).fit(disp=0)
    lr_stat = 2 * (sp.llf - base.llf)
    df_diff = sp.df_model - base.df_model
    print(f"{v:15s} LR stat = {lr_stat:7.2f}, df = {df_diff:.0f}, p = {stats.chi2.sf(lr_stat, df_diff):.4g}")


# In[20]:

# Practical cost of non-linearity
nonlinear = ['tenure', 'TotalCharges']   
lin_pipe = make_pipeline(
    MinMaxScaler(),
    LogisticRegression(C=best_C, max_iter=10000, random_state=42)
)
spline_pipe = make_pipeline(
    ColumnTransformer(
        [('spl', SplineTransformer(n_knots=5, degree=3, include_bias=False), nonlinear)],
        remainder=MinMaxScaler()
    ),
    LogisticRegression(C=best_C, max_iter=10000, random_state=42)
)

for name, pipe in [('Linear logit', lin_pipe), ('Spline logit', spline_pipe)]:
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv5, method='predict_proba')[:, 1]
    print(f"{name:13s} PR-AUC={average_precision_score(y_train, oof):.4f}  "
          f"ROC-AUC={roc_auc_score(y_train, oof):.4f}  log-loss={log_loss(y_train, oof):.4f}")


print("#--"*10)
print("\n Model Diagnostics: ROC Curve & Threshold Analysis:")

# In[21]:

# Predict probabilities for the positive class using the fitted KNN model
y_prob_knn = knn_model.predict_proba(X_test_scaled)[:, 1]
print(knn_model.classes_)   #  check if col 1 = churn


# In[22]:

# Predict probabilities for the positive class using the fitted Logistic Regression model
y_prob_logreg = log_model.predict_proba(X_test_scaled)[:, 1]

print("#--"*10)
print("\n Probability Calibration:")
# The logistic regression is fit without class weights, so its outputs estimate P(churn | x) on the natural scale. Class imbalance is handled at the decision stage by choosing a threshold, not by resampling or reweighting the training data (van den Goorbergh et al., 2022). Calibration is verified on the held-out test set using a reliability diagram, calibration slope and intercept, and the Brier score.

# In[23]:

# Calibration check
p = np.clip(y_prob_logreg, 1e-6, 1 - 1e-6) # the logit is infinite at 0 and 1
lp = logit(p) # converting clipped probabilities to log-odds

# Calibration slope (ideal would be 1)
slope_fit = sm.Logit(y_test, sm.add_constant(lp)).fit(disp=0) # disp=0 : no convergence details
cal_slope = slope_fit.params.iloc[1]
cal_slope_ci = slope_fit.conf_int().iloc[1].values

# Calibration-in-the-large (estimate intercept with slope fixed at 1; ideal = 0)
citl_fit = sm.GLM(y_test, np.ones(len(lp)), family=sm.families.Binomial(),
                  offset=lp).fit() 
cal_citl = citl_fit.params.iloc[0]

brier = brier_score_loss(y_test, p) # standard Brier score loss function from scikit learn
brier_ref = y_test.mean() * (1 - y_test.mean())   # constant-prediction Brier
print(f"Test churn rate:      {y_test.mean():.3f}")
print(f"Mean predicted prob:  {p.mean():.3f}")
print(f"Brier score:          {brier:.4f}  (constant-prediction reference {brier_ref:.4f})")
print(f"Calibration slope:    {cal_slope:.2f}  (95% CI {cal_slope_ci[0]:.2f}-{cal_slope_ci[1]:.2f})")
print(f"Calibration intercept:{cal_citl:+.2f}")

# Reliability diagram
frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy='quantile')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(mean_pred, frac_pos, marker='o', color='purple', label='Logistic Regression')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Observed churn rate')
ax.set_title('Reliability diagram (test set, 10 quantile bins)')
ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('images/reliability_diagram.png', dpi=300, bbox_inches='tight')
plt.show()


print("#--"*10)
print("\n Thresholds:")
# 
# A threshold is the cut-off applied to a predicted probability to assign the positive class — in this context, the probability above which a customer is flagged as at-risk of churn. Threshold choice directly trades off true positives against false positives.
# 
# Rather than evaluating a static grid of thresholds, the ROC curve sweeps every unique predicted probability produced by the model. This gives an exact curve with no redundant computation, since each unique probability represents a distinct point where the model's classifications change.

# In[24]:

def get_thresholds(y_pred_proba):
    """Return sorted unique thresholds, padded to force ROC endpoints."""
    unique_probas = np.unique(y_pred_proba)[::-1]

    # Insert 1.1 → forces (FPR=0, TPR=0), the lower-left corner
    # Append 0  → forces (FPR=1, TPR=1), the upper-right corner
    thresholds = np.insert(unique_probas, 0, 1.1)
    thresholds = np.append(thresholds, 0)
    return thresholds

# Manual implementation of threshold-based FPR/TPR, equivalent to sklearn.metrics.roc_curve


# In[25]:

# Generate threshold arrays for ROC evaluation
knn_thresholds = get_thresholds(y_prob_knn)
logreg_thresholds = get_thresholds(y_prob_logreg)

# Sanity check
print(f"KNN thresholds: {len(knn_thresholds)}  |  LR thresholds: {len(logreg_thresholds)}")

print("FPR & TPR")

# In[26]:

def get_fpr(y_true, y_pred_proba, threshold):
    """False positive rate at each threshold. Returns scalar if input is scalar."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    scalar_input = isinstance(threshold, (int, float, np.number))
    thresholds = np.atleast_1d(threshold)

    total_negatives = (y_true == 0).sum()
    preds = y_pred_proba[:, None] >= thresholds[None, :]
    fp = (preds & (y_true[:, None] == 0)).sum(axis=0)

    fpr = fp / total_negatives
    return fpr[0] if scalar_input else fpr

# In[27]:

def get_tpr(y_true, y_pred_proba, threshold):
    """True positive rate at each threshold. Returns scalar if input is scalar."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    scalar_input = isinstance(threshold, (int, float, np.number))
    thresholds = np.atleast_1d(threshold)

    total_positives = (y_true == 1).sum()
    preds = y_pred_proba[:, None] >= thresholds[None, :]
    tp = (preds & (y_true[:, None] == 1)).sum(axis=0)

    tpr = tp / total_positives
    return tpr[0] if scalar_input else tpr

# In[28]:

# FPR for the kNN at each of its thresholds
knn_fpr = get_fpr(y_test, y_prob_knn, knn_thresholds)
# TPR for the kNN at each of its thresholds
knn_tpr = get_tpr(y_test, y_prob_knn, knn_thresholds)

# TPR for the logistic model at each of its thresholds
logreg_tpr = get_tpr(y_test, y_prob_logreg, logreg_thresholds)
# FPR for the logistic model at each of its thresholds
logreg_fpr = get_fpr(y_test, y_prob_logreg, logreg_thresholds)

print("#--"*10)
print("\n Area Under the Curve (AUC):")
# 
# - The AUC gives us an idea as to how well the model does across thresholds.
# - For our final calculations, use each model's predicted probabilities to compute its test AUC with the help of SKLearn's `roc_auc_score` function.

# In[29]:

# Compute the ROC AUC score of the kNN model
knn_auc = roc_auc_score(y_test, y_prob_knn)

# Compute the ROC AUC score of the Logistic model 
logreg_auc = roc_auc_score(y_test, y_prob_logreg)

print(f"KNN AUC:      {knn_auc:.4f}")
print(f"LR AUC:       {logreg_auc:.4f}")


# In[30]:

fig, ax = plt.subplots(figsize=(12, 8))

# Plot KNN Classifier ROC Curve
ax.plot(knn_fpr, knn_tpr,
        label=f'KNN Classifier (AUC = {knn_auc:.3f})',
        color='g', lw=3)

# Plot Logistic Regression ROC Curve
ax.plot(logreg_fpr, logreg_tpr,
        label=f'Logistic Regression (AUC = {logreg_auc:.3f})',
        color='purple', lw=3)

label_kwargs = {'bbox': dict(boxstyle='round, pad=0.2',
                             facecolor='white', edgecolor='lightgray', alpha=0.9)}
eps = 0.03  # vertical offset

# Annotate LR thresholds
for i in range(0, len(logreg_fpr), 200):
    if i < len(logreg_thresholds):
        ax.annotate(f"{logreg_thresholds[i]:.2f}",
                    (logreg_fpr[i], logreg_tpr[i] + eps),
                    fontsize=9, color='purple', weight='bold', **label_kwargs)

# Annotate KNN thresholds
for i in range(0, len(knn_fpr), 150):
    if i < len(knn_thresholds):
        ax.annotate(f"{knn_thresholds[i]:.2f}",
                    (knn_fpr[i], knn_tpr[i] - eps),
                    fontsize=9, color='green', weight='bold', **label_kwargs)

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess Baseline (AUC = 0.500)')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=16, labelpad=10)
ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=16, labelpad=10)
ax.set_title('ROC Curve & Threshold Diagnostics', fontsize=18, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=13, frameon=True, facecolor='white', edgecolor='lightgray')
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('images/roc_curve_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


print("#--"*10)
print("\n Model Diagnostics: Precision-Recall Curve:")
# ---  
# The Precision-Recall (PR) curve plots precision against recall at every decision threshold. Unlike ROC — which uses the false positive rate over *all* negatives — PR's precision denominator is only the *predicted positives*. On imbalanced data (26.5% churn), this makes PR a more honest view of a classifier's positive-class performance.
# 
# The no-skill baseline is a horizontal line at the class rate (**0.265**). A model that performs above this baseline has real predictive signal.

# In[31]:

# Compute PR curves
knn_precision, knn_recall, _ = precision_recall_curve(y_test, y_prob_knn)
log_precision, log_recall, _ = precision_recall_curve(y_test, y_prob_logreg)

# PR-AUC scores
knn_pr_auc = average_precision_score(y_test, y_prob_knn)
log_pr_auc = average_precision_score(y_test, y_prob_logreg)

fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(knn_recall, knn_precision, color='g', lw=3,
        label=f'KNN (PR-AUC = {knn_pr_auc:.3f})')
ax.plot(log_recall, log_precision, color='purple', lw=3,
        label=f'Logistic Regression (PR-AUC = {log_pr_auc:.3f})')

# Baseline: no-skill classifier = churn rate
baseline = y_test.mean()
ax.axhline(baseline, color='k', linestyle='--', lw=1.5,
           label=f'No-Skill Baseline ({baseline:.3f})')

ax.set_xlabel('Recall (Sensitivity)', fontsize=14)
ax.set_ylabel('Precision', fontsize=14)
ax.set_title('Precision-Recall Curve', fontsize=16, fontweight='bold')
ax.set_xlim([0, 1.02])
ax.set_ylim([0, 1.02])
ax.legend(loc='lower left', fontsize=12, frameon=True, facecolor='white', edgecolor='lightgray')
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('images/precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"KNN PR-AUC:  {knn_pr_auc:.4f}")
print(f"LR PR-AUC:   {log_pr_auc:.4f}")
print(f"Baseline:    {baseline:.4f}")

print("#--"*10)
print("\nStatistical Significance Testing (Bootstrap & McNemar) - Model Comparison:")

# In[32]:

rng = np.random.default_rng(42)
y = np.asarray(y_test)
n = len(y)
diff_roc, diff_pr = [], [] # ROC-AUC and PR-AUC
for _ in range(2000): # 2000 bootsrap samples
    idx = rng.integers(0, n, n) # random pick with replacement
    diff_roc.append(roc_auc_score(y[idx], y_prob_log[idx]) - roc_auc_score(y[idx], y_prob_knn[idx]))
    diff_pr.append(average_precision_score(y[idx], y_prob_log[idx]) - average_precision_score(y[idx], y_prob_knn[idx]))
for name, d in [('ROC-AUC', diff_roc), ('PR-AUC', diff_pr)]:
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"LR − KNN {name}: {np.mean(d):+.3f}  (95% bootstrap CI {lo:+.3f} to {hi:+.3f})")

ok_l = (y_pred_log == y)
ok_k = (y_pred_knn == y)
tbl = [[(ok_l & ok_k).sum(), (ok_l & ~ok_k).sum()],
       [(~ok_l & ok_k).sum(), (~ok_l & ~ok_k).sum()]]
print("McNemar (LR vs KNN correctness):", mcnemar(tbl, exact=False, correction=True))


print("#--"*10)
print("\n Data-Driven Strategic Recommendations 📊:")

print("#--"*10)
print("\n Objective 4:")
print("\nHow can predicted churn probabilities be used to identify high-risk customers for targeted retention efforts?")
# 
# ---

print("#--"*10)
print("\n Simulate the Risk Tranches:")
# 
# The fitted Logistic Regression model outputs a **continuous churn probability** for every customer in the test set — a value between 0 and 1. On its own, that number isn't actionable: a retention team can't make a business decision from a value like "0.63." What they need is a **prioritized list** of customers grouped by how urgently they should be contacted.
# 
# This section bridges that gap by slicing the test population into **four risk tranches** — Critical, High, Medium, and Low — based on probability cut-offs at 0.75, 0.50, and 0.25. Each tranche corresponds to a distinct intervention strategy: immediate high-value offers for Critical, targeted retention for High, proactive engagement for Medium, and no action for Low.
# 
# The resulting **simulated dataframe** (`sim_df`) carries only the fields needed for the downstream financial simulation:
# 
# - `tenure`, `MonthlyCharges` — the two revenue-relevant features used for context and billing math.
# - **`Churn_Probability`** — the model's output, which already encodes everything the classifier learned from all 13 selected features. It is the sole driver of tranche assignment.
# - **`Actual_Churn`** — the ground-truth label, retained for later validation of whether high-probability tranches really do churn more often.
# 
# The remaining 11 features are deliberately excluded. They were used to *train* the model, but they add nothing to the simulation: the model's learned signal is fully captured in `Churn_Probability`, and the only other input the ROI calculation needs is `MonthlyCharges`.
# 
# > **Note:** Because `Churn_Probability` was produced from the full 13-feature scaled test set, this reduced frame cannot regenerate the model output on its own. It is a downstream artifact of the modeling pipeline, not a standalone feature table.

# In[33]:

# Build the simulation dataframe
sim_df = pd.DataFrame({
    'tenure': X_test['tenure'],
    'MonthlyCharges': X_test['MonthlyCharges'],
    'Churn_Probability': y_prob_logreg,
    'Actual_Churn': y_test,
})

# Business-defined risk tranches via pd.cut
sim_df['Risk_Segment'] = pd.cut(
    sim_df['Churn_Probability'],
    bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
    labels=['4. Low Risk', '3. Medium Risk', '2. High Risk', '1. Critical Risk'],
    right=False,
)

# Safety check: any customer got dropped during the pd.cut?
assert sim_df['Risk_Segment'].notna().all(), "Unclassified customers detected"


# Summarize each tranche: volume, average bill, and probability-weighted exposure.
sim_df['Expected_Revenue_At_Risk'] = sim_df['MonthlyCharges'] * sim_df['Churn_Probability']

summary = sim_df.groupby('Risk_Segment', observed=True).agg(
    Customer_Count=('Churn_Probability', 'count'),
    Avg_Monthly_Bill=('MonthlyCharges', 'mean'),
    Expected_Monthly_Revenue_At_Risk=('Expected_Revenue_At_Risk', 'sum'),
    Observed_Churn_Rate=('Actual_Churn', 'mean'),
)

summary[['Avg_Monthly_Bill', 'Expected_Monthly_Revenue_At_Risk']] = \
    summary[['Avg_Monthly_Bill', 'Expected_Monthly_Revenue_At_Risk']].round(2)
summary['Observed_Churn_Rate'] = summary['Observed_Churn_Rate'].round(3)

print(summary)


# In[34]:

# --- The risk-tranche summary --- 
total_expected = summary['Expected_Monthly_Revenue_At_Risk'].sum()
top2_expected  = summary.loc[['1. Critical Risk', '2. High Risk'],
                             'Expected_Monthly_Revenue_At_Risk'].sum()

total_customers = summary['Customer_Count'].sum()
top2_customers  = summary.loc[['1. Critical Risk', '2. High Risk'],
                              'Customer_Count'].sum()

# Billing gap between highest and lowest tranche
bill_gap = (summary.loc['1. Critical Risk', 'Avg_Monthly_Bill']
            - summary.loc['4. Low Risk', 'Avg_Monthly_Bill'])

# Tranche-level churn-rate spread
churn_spread = (summary.loc['1. Critical Risk', 'Observed_Churn_Rate']
                - summary.loc['4. Low Risk', 'Observed_Churn_Rate'])

print(f"Total expected monthly revenue at risk:        ${total_expected:,.2f}")
print(f"Top-2 expected monthly revenue at risk:        ${top2_expected:,.2f}")
print(f"Top-2 share of total expected revenue:         {top2_expected/total_expected:.1%}")
print()
print(f"Total customers in test set:                   {total_customers:,}")
print(f"Top-2 customer count:                          {top2_customers:,}")
print(f"Top-2 share of customer base:                  {top2_customers/total_customers:.1%}")
print()
print(f"Avg bill gap (Critical − Low):                 ${bill_gap:,.2f}")
print(f"Observed churn rate spread (Critical − Low):   {churn_spread:.1%} "
      f"({summary.loc['1. Critical Risk','Observed_Churn_Rate']:.1%} vs "
      f"{summary.loc['4. Low Risk','Observed_Churn_Rate']:.1%})")


print("#--"*10)
print("\n Objective 5:")
print(" What is the estimated financial impact of targeting high-risk customers with a retention intervention?")
# ---
print("#--"*10)
print("\n Financial Impact Simulation:")
# To estimate the financial impact, the simulation applies a deterministic cost-benefit model to the identified high-risk tranches. It explicitly relies on the following assumptions:
# - **Campaign Scope:** All **Critical** and **High** risk customers are targeted.
# - **Base Success Rate:** It is assumed that the intervention saves **20%** of the targeted churners. This baseline is set from practitioner experience & knowledge with medium-touch retention interventions, rather than an empirically calibrated estimate from a published telecom trial (see Limitations). However, because true efficacy is uncertain, a sensitivity analysis across 15%–50% is tested later to show how the ROI scales.
# - **Offer Cost:** A range of monthly bill discounts (\\$2 to \\$15) were tested.
# - **Time Horizon:** Financial impact is calculated over a 12-month period.
# - **Upper-Bound Issue:** Saves are counted only among observed churners, but the model assumes every saved customer stays the full 12 months, counts revenue rather than margin, and applies one flat success rate to every offer level. The results should be treated as an upper bound.

# In[35]:

# =============================================================
# Baseline parameters
# =============================================================
print("Size of test sample:", len(y_test), "\n")
# The target group (Critical + High risk)
targeted = sim_df[sim_df['Churn_Probability'] >= 0.50]

# Actual churners in this group
churners = targeted[targeted['Actual_Churn'] == 1]
print("Actual churners in the targeted sample", len(churners), "\n")
print("Churners in the full test sample:", y_test.sum(), "\n")

# Calculate exact metrics from the raw data
total_targeted_customers = len(targeted)
actual_churners_targeted = len(churners)

# The average bill of the churners
avg_bill_churners = churners['MonthlyCharges'].mean()

# =============================================================
# Assumptions
# =============================================================
intervention_success_rate = 0.20
customer_lifespan_months = 12

# Derived metrics
customers_saved = round(actual_churners_targeted * intervention_success_rate)
gross_revenue_saved = customers_saved * avg_bill_churners * customer_lifespan_months

print(f"Campaign recall: {actual_churners_targeted / y_test.sum():.1%} of all test churners "
      f"({actual_churners_targeted} of {int(y_test.sum())}) fall in the targeted tranches")

# =============================================================
# Sensitivity Analysis: monthly_retention_offer_cost
# =============================================================
offer_costs = [2, 5, 7, 10, 15]

roi_table = PrettyTable()
roi_table.field_names = ["Offer $/mo", "Campaign Cost", "Gross Revenue Saved", "Net Benefit", "ROI"]
roi_table.align["Offer $/mo"] = "l"  
roi_table.align = "r"               

for cost in offer_costs:
    # Assume the offer is extended to all targeted accounts for the full 12 months
    total_campaign_cost = total_targeted_customers * cost * customer_lifespan_months
    net_revenue_saved   = gross_revenue_saved - total_campaign_cost
    roi_pct             = (net_revenue_saved / total_campaign_cost) * 100

    roi_table.add_row([
        f"${cost:,}",
        f"USD {total_campaign_cost:,.0f}", 
        f"USD {gross_revenue_saved:,.0f}",
        f"USD {net_revenue_saved:,.0f}",
        f"{roi_pct:.1f}%",
    ])

# =============================================================
# Display
# =============================================================
print("=" * 80)
print("      RETENTION CAMPAIGN ROI — SENSITIVITY TO MONTHLY OFFER COST")
print("=" * 80)
print(f"Targeted customers: {total_targeted_customers:,}  |  "
      f"Churners (observed): {round(actual_churners_targeted):,}  |  "
      f"Saved @ {int(intervention_success_rate*100)}%: {customers_saved:,}")
print(f"Avg bill of churners: ${avg_bill_churners:,.2f}/mo  |  "
      f"Gross revenue protected (12 mo): USD {gross_revenue_saved:,.2f}")
print("-" * 80)
print(roi_table)


print("#--"*10)
print("\n Scenarios for Different Intervention Success Rates:")

# In[36]:

success_rates = [0.15, 0.25, 0.40, 0.50]

# Fixed width for the offer columns
COL_WIDTH = 10

header = "Success Rate | " + " | ".join(
    [f"{'Offer $' + str(cost):>{COL_WIDTH}}" for cost in offer_costs]
)
print(header)
print("-" * len(header))

for sr in success_rates:
    customers_saved_sr = round(actual_churners_targeted * sr)
    gross_saved_sr = customers_saved_sr * avg_bill_churners * customer_lifespan_months

    row = []
    for cost in offer_costs:
        cc = total_targeted_customers * cost * customer_lifespan_months
        row.append(f"{((gross_saved_sr - cc) / cc) * 100:,.1f}%")

    print(f"{sr:>12.0%} | " + " | ".join(f"{val:>{COL_WIDTH}}" for val in row))


# #### Visualizing ROIs for the Monthly Discounts

# In[37]:

rois = [(gross_revenue_saved - total_targeted_customers * c * 12) /
        (total_targeted_customers * c * 12) * 100 for c in offer_costs]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(offer_costs, rois, marker='o', color='purple', lw=2)
ax.axhline(0, color='black', ls='--', lw=1, label='Break-even')
ax.set_xlabel('Monthly retention offer cost (USD)')
ax.set_ylabel('ROI (%)')
ax.set_title('ROI vs. Monthly Offer Cost')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()
plt.tight_layout()
plt.savefig('images/roi_vs_offer_cost.png', dpi=300, bbox_inches='tight')
plt.show()

# In[38]:

# =============================================================
# Break-even success rate (SR) by offer size
# =============================================================
# ROI = (Gross - Cost) / Cost = 0  =>  Gross = Cost
# Gross = (Churners * SR) * AvgBill * Months
# Cost  = Targeted * OfferCost * Months
#
# Set Gross = Cost and solve for SR:
# SR = (Targeted * OfferCost) / (Churners * AvgBill)
#
# Note: customer_lifespan_months cancels out of the equation.

print(f"{'Offer $/mo':<12} | {'Break-even Success Rate':>24}")
print("-" * 39)

for cost in offer_costs:
    # Target gross revenue must equal target campaign cost
    required_gross = total_targeted_customers * cost

    # Max possible gross (if we saved 100% of churners)
    max_gross = actual_churners_targeted * avg_bill_churners

    # Required success rate to break even
    breakeven_sr = required_gross / max_gross

    print(f"${cost:<11} | {breakeven_sr:>23.1%}")


# Orhan Kaplan  
# Customer Churn Prediction and Retention Strategy Analysis with Business Impact Simulation
# GitHub URL: https://github.com/nalpako2027/telco_customer_churn_prediction_simulation.git





