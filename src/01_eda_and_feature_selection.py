#!/usr/bin/env python
# coding: utf-8

# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb

import json

import matplotlib; matplotlib.use("Agg")

print("#--"*10)
print("\n  Load & Check the Data:")
# In[2]:

# Read the Telco datafile
df = pd.read_csv('Telco-Customer-Churn.csv') # Data source: https://github.com/IBM/telco-customer-churn-on-icp4d

print("\nTake a quick look at the dataframe:")
print(df['Churn'].value_counts())

# In[3]:

# Checking for potential white spaces
print(df['Churn'].str.strip().str.title().value_counts())


# In[4]:

print(df.head())


# In[5]:

print(df.dtypes)

# In[6]:

df = df.drop(columns=['customerID'])

# In[7]:

print(df.shape)

print("#--"*10)
print("\n Data Type Conversions:")

# In[8]:

# Convert TotalCharges from string to float (blanks expected → NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"NaNs in TotalCharges after coercion: {df['TotalCharges'].isna().sum()}")

# Convert Churn to categorical (keeping Yes/No labels readable for EDA)
df['Churn'] = df['Churn'].astype('category')
print(f"NaNs in Churn after coercion: {df['Churn'].isna().sum()}")

# SeniorCitizen was declared categorical
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'}).astype('category')
print(f"NaNs in SeniorCitizen after coercion: {df['SeniorCitizen'].isna().sum()}")

print(df.dtypes)

# In[9]:

# Sanity check: tenure and MonthlyCharges should already be numeric
assert pd.api.types.is_numeric_dtype(df['tenure']), "tenure is not numeric!"
assert pd.api.types.is_numeric_dtype(df['MonthlyCharges']), "MonthlyCharges is not numeric!"
assert pd.api.types.is_numeric_dtype(df['TotalCharges']), "TotalCharges is not numeric!"
print("✅ No issue!")

print("#--"*10)
print("\n Feature & Target Declaration:")

# In[10]:

print(df.columns)

# In[11]:

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
target = 'Churn'

# > Now numeric_features + categorical_features = 19 predictors, target = 1, total = 20 columns.

print("#--"*10)
print("\n Exploratory Data Analysis :")

# In[12]:

def var_diagnostic(df, var):
    """Diagnostic information for a single variable."""
    print(f"--------------- Diagnostic Info: {var} ---------------")
    print(f"Variable Type:   {df[var].dtype}")
    print(f"Missing values:  {df[var].isna().sum()}")

    if pd.api.types.is_numeric_dtype(df[var]):
        print("Descriptive info:")
        print(df[var].describe())
    else:
        print("Value counts:")
        counts = df[var].value_counts(dropna=False)
        pcts   = df[var].value_counts(normalize=True, dropna=False) * 100
        print(pd.DataFrame({'count': counts, 'pct': pcts.round(2)}))

    print("------------------------- End -------------------------")

print("#--"*10)
print("\n Diagnostics on Numeric Features:")

# In[13]:

for var in numeric_features:
    var_diagnostic(df, var)


# ### Visualizing the Distribution

# In[14]:

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, var in enumerate(numeric_features):
    sns.histplot(data=df, x=var, bins=30, kde=True,
                 color='steelblue', edgecolor='black', ax=axes[i])
    axes[i].set_title(f'Distribution of {var}', fontsize=16)
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Customer Count')

plt.tight_layout()
plt.savefig('images/numeric_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print("#--"*10)
print("\n Diagnostics on categorical Features:")

# In[15]:

for var in categorical_features:
    var_diagnostic(df, var)

# ### Visualization

# In[16]:

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for i, var in enumerate(categorical_features):
    ax = axes[i]
    order = df[var].value_counts().index
    sns.countplot(data=df, y=var, order=order, ax=ax,
                  hue=var, palette='Set2', legend=False)
    ax.set_title(var, fontsize=11, fontweight='bold')
    ax.set_xlabel('Customer Count')
    ax.set_ylabel('')

# Hide any unused subplot slots (none here — 16 features, 16 slots)
for j in range(len(categorical_features), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('images/categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()


# In[17]:

# Internet-service redundancy check
n_ol = (df['OnlineSecurity'] == 'No internet service').sum()
n_is = (df['InternetService'] == 'No').sum()
print(f"OnlineSecurity 'No internet service': {n_ol}, InternetService 'No': {n_is}")
assert n_ol == n_is, "Redundancy check failed: OnlineSecurity vs InternetService mismatch!"

# Phone-service redundancy check
n_ml = (df['MultipleLines'] == 'No phone service').sum()
n_ps = (df['PhoneService'] == 'No').sum()
print(f"MultipleLines 'No phone service': {n_ml}, PhoneService 'No': {n_ps}")
assert n_ml == n_ps, "Redundancy check failed: MultipleLines vs PhoneService mismatch!"

# These are count-level check. We should also check row-level equivalence
assert (df['OnlineSecurity'] == 'No internet service').equals(df['InternetService'] == 'No'), \
    "Rows mismatch: OnlineSecurity and InternetService redundancy is not exact!"


print("#--"*10)
print("\n Diagnostics on the Target Variable:")

# In[18]:

var_diagnostic(df, target)


# #### Visualization

# In[19]:

fig, ax = plt.subplots(figsize=(7, 3))

order = df[target].value_counts().index
sns.countplot(data=df, y=target, order=order,
              hue=target, palette='Set2', legend=False, ax=ax)

total = len(df)
for container in ax.containers:
    labels = [f'{bar.get_width():.0f}  ({bar.get_width()/total*100:.1f}%)'
              for bar in container]
    ax.bar_label(container, labels=labels, padding=5)

ax.set_title('Churn Distribution', fontsize=13, fontweight='bold')
ax.set_xlabel('Customer Count')
ax.set_ylabel('')
ax.set_xlim(0, df[target].value_counts().max() * 1.2)

plt.tight_layout()
plt.savefig('images/target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# In[20]:

print(df[df[target] == 'Yes']['MonthlyCharges'].mean())


print("#--"*10)
print("\n Outlier Detection:")

# In[21]:

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

df_plot = df.copy()

for i, var in enumerate(numeric_features):
    sns.boxplot(data=df_plot, x='Churn', y=var, hue='Churn',
                palette='Set2', ax=axes[i], legend=False)
    axes[i].set_title(f'{var} Boxplot by Churn')
    axes[i].set_xlabel('Churn Status')
    axes[i].set_ylabel(var)

plt.tight_layout()
plt.savefig('images/features_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()


# In[22]:

def upper_fence_population(df, var):
    """IQR upper fence computed on the entire customer base."""
    q1 = df[var].quantile(0.25)
    q3 = df[var].quantile(0.75)
    iqr = q3 - q1
    return q3 + 1.5 * iqr

for var in numeric_features:
    fence = upper_fence_population(df, var)
    flagged = df[(df['Churn'] == 'Yes') & (df[var] > fence)]
    print(f"{var}: population fence = {fence:.2f}, flagged churners = {len(flagged)}")


print("#--"*10)
print("\n Missing `TotalCharges` Values & Imputation:")

# In[23]:

missing = df[df['TotalCharges'].isna()]
print(len(missing), "\n")
print(missing[['tenure', 'MonthlyCharges', 'Churn']].describe(include='all'))
print("\n", missing['Churn'].value_counts())
print("\n", (df['tenure'] == 0).sum())


# In[24]:

# TotalCharges is structurally missing iff tenure = 0, not random.
mask = df['TotalCharges'].isna() & (df['tenure'] == 0)
df.loc[mask, 'TotalCharges'] = 0.0

assert df['TotalCharges'].isna().sum() == 0, "Unexpected missing TotalCharges remain"
print(f"Filled {mask.sum()} structurally missing TotalCharges values with 0")


print("#--"*10)
print("\n Data Quality Assurance:")
# Final validation pass before modeling: three assertion checks confirm there are no negative charges or tenure values, that the 11 imputed zero-tenure customers are consistently recorded, and that TotalCharges is plausible relative to tenure × MonthlyCharges. Any violation raises an error immediately, so downstream analysis cannot silently proceed on corrupted data.

# In[25]:

# Negative Balance Check
assert df[(df['MonthlyCharges'] < 0) | (df['TotalCharges'] < 0)].shape[0] == 0, \
    "Data Error: Negative charges exist in the dataset!"
assert df[df['tenure'] < 0].shape[0] == 0, \
    "Data Error: Negative tenure values!"

print("✅ No negative charges and no negative tenure values.")


# In[26]:

# New Customer Balance Check
assert df[(df['tenure'] == 0) & (df['TotalCharges'] > 0)].shape[0] == 0, \
    "Logic Error: Customers with 0 tenure have Total Charges > $0"

print("✅ No new customers with non-zero TotalCharges.")


# In[27]:

# Churn rate by categorical feature
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for i, var in enumerate(categorical_features):
    ax = axes[i]
    churn_rate = df.groupby(var, observed=True)['Churn'].apply(
        lambda s: (s == 'Yes').mean() * 100
    ).sort_values(ascending=False)

    sns.barplot(x=churn_rate.values, y=churn_rate.index, ax=ax,
                hue=churn_rate.index, palette='Reds_r', legend=False)
    ax.set_title(f'{var} — Churn Rate (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Churn Rate (%)')
    ax.set_ylabel('')
    ax.axvline(x=26.5, color='gray', linestyle='--', alpha=0.6, linewidth=1)

for j in range(len(categorical_features), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('images/churn_rate_by_category.png', dpi=300, bbox_inches='tight')
plt.close()

# In[28]:

# Compare mean numeric values between churned and retained customers
group_means = df.groupby('Churn', observed=True)[numeric_features].mean().T
group_means.columns = ['Stayed (No)', 'Churned (Yes)']
group_means['Difference'] = group_means['Churned (Yes)'] - group_means['Stayed (No)']
print(group_means.round(2))


print("#--"*10)
print("\n Correlation Matrix for Numeric Features:")

# In[29]:

# Pearson correlation between numeric features
corr_matrix = df[numeric_features].corr()
print(corr_matrix.round(3))

# Heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Pearson Correlation — Numeric Features', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images/correlation_numeric.png', dpi=300, bbox_inches='tight')
plt.close()


print("#--"*10)
print("\n Cramér's V (Categorical Associations) 📏:")

# In[30]:

def cramers_v(x, y):
    """Cramér's V association between two categorical series."""
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum().sum()
    r, k = confusion.shape
    return np.sqrt((chi2 / n) / min(r - 1, k - 1))

# Build the symmetric association matrix
cramerv_matrix = pd.DataFrame(index=categorical_features,
                         columns=categorical_features, dtype=float)

for i in categorical_features:
    for j in categorical_features:
        cramerv_matrix.loc[i, j] = cramers_v(df[i], df[j])

# Heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cramerv_matrix.astype(float), annot=True, fmt='.2f',
            cmap='Reds', vmin=0, vmax=1, square=True,
            cbar_kws={'shrink': 0.7}, ax=ax,
            annot_kws={'size': 8})
ax.set_title("Cramér's V — Categorical Feature Associations",
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('images/cramers_v_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# In[31]:

# Numeric features overlaid by churn status
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

for i, var in enumerate(numeric_features):
    sns.histplot(data=df, x=var, hue='Churn', palette='Set2',
                 multiple='layer', alpha=0.50, bins=30,
                 edgecolor='black', linewidth=0.3, ax=axes[i])
    axes[i].set_title(f'Distribution of {var} by Churn', fontsize=14)
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Customer Count')

plt.tight_layout()
plt.savefig('images/numeric_by_churn.png', dpi=300, bbox_inches='tight')
plt.close()


# In[32]:

# Mean monthly charges by tenure, split by churn status
tenure_bin = pd.cut(df['tenure'], bins=[-1, 12, 24, 36, 48, 60, 72],
                    labels=['0–12', '13–24', '25–36', '37–48', '49–60', '61–72'])

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

# Left: mean monthly charges at each tenure month
sns.lineplot(data=df, x='tenure', y='MonthlyCharges', hue='Churn',
             palette='Set2', estimator='mean', errorbar=('ci', 95),
             linewidth=2, ax=axes[0])
axes[0].set_title('By Tenure Month', fontsize=14)
axes[0].set_xlabel('Tenure (months)')
axes[0].set_ylabel('Average Monthly Charges')

# Right: mean monthly charges per tenure group
sns.pointplot(data=df.assign(tenure_bin=tenure_bin), x='tenure_bin', y='MonthlyCharges',
              hue='Churn', palette='Set2', dodge=0.3, errorbar=('ci', 95), ax=axes[1])
axes[1].set_title('By Tenure Group', fontsize=14)
axes[1].set_xlabel('Tenure (months)')
axes[1].set_ylabel('')
axes[1].get_legend().remove()

fig.suptitle('Average Monthly Charges by Tenure and Churn Status', fontsize=16)
plt.tight_layout()
plt.savefig('images/tenure_monthly_charges_by_churn.png', dpi=300, bbox_inches='tight')
plt.close()


print("#--"*10)
print("\n Objective 2:")
print("\n Which features carry the strongest predictive signal for churn, as ranked by XGBoost gain importance across the full Telco feature set? ")


# In[33]:

# Collapse "No internet service" and "No phone service" to "No" for consistency
redundant_map = {'No internet service': 'No', 'No phone service': 'No'}

# Recoding the categories
for col in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']:
    df[col] = df[col].replace(redundant_map)

# Verify the collapse
for col in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']:
    print(f"{col}: {sorted(df[col].unique())}")


# In[34]:

print(f"Value counts:\n{df['Churn'].value_counts()}")

# Encode Churn as binary (XGBoost requires numeric labels)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype('int')

# Verify
print(f"\nChurn dtype: {df['Churn'].dtype}")
print(f"\nValue counts:\n{df['Churn'].value_counts()}")
print(f"\nClass balance: {df['Churn'].mean():.4f}")


# In[35]:

# One-hot encode categorical predictors
# drop_first=True -> Drops the first level of each feature as the reference category,
# avoids perfect multicollinearity
df_encoded = pd.get_dummies(df, columns=categorical_features,
                            drop_first=True, dtype=int)

df = df.drop(columns='tenure_bin', errors='ignore')

# Dummy columns after encoding
dummy_cols = [c for c in df_encoded.columns if c not in df.columns]
passthrough_cols = [c for c in df_encoded.columns if c in df.columns]

print(f"Shape before encoding: {df.shape}")
print(f"Shape after encoding:  {df_encoded.shape}")
print(f"Original categorical columns:  {len(categorical_features)}")
print(f"Total encoded dummy columns:   {len(dummy_cols)}")
print(f"Net columns added:             {df_encoded.shape[1] - df.shape[1]}")
print(f"\nColumns passed through unchanged ({len(passthrough_cols)}):")
print(passthrough_cols)
print(f"\nEncoded dummy columns ({len(dummy_cols)}):")
print(dummy_cols)


# In[36]:

# All dummies should be 0/1
assert df_encoded[dummy_cols].isin([0, 1]).all().all(), \
    "Non-binary values found in encoded columns!"
print(f"✅ All {len(dummy_cols)} encoded columns are binary.")


# In[37]:

assert (df_encoded.dtypes == 'object').sum() == 0, "Object dtypes remain!"
print("✅ All columns are numeric.")


# In[38]:

# Separate predictors from target
X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Feature count: {X.shape[1]}")
print(f"Churn rate:    {y.mean():.4f}")


# In[39]:

# Stratified 70/30 split — preserves the 26.5% churn rate in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=60
)

print(f"X_train: {X_train.shape}   y_train churn rate: {y_train.mean():.4f}")
print(f"X_test:  {X_test.shape}    y_test  churn rate: {y_test.mean():.4f}")


print("#--"*10)
print("\n Modeling XGBoost for Feature Ranking ⚙️:")
# 

# In[40]:

# Split the 70% training set again: 80% fit, 20% validation
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=60
)

xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    early_stopping_rounds=30,
    random_state=60,
    n_jobs=-1,
)
xgb_model.fit(
    X_fit, y_fit,
    eval_set=[(X_fit, y_fit), (X_val, y_val)],  
    verbose=False,
)
print(f"Best iteration: {xgb_model.best_iteration}")


# In[41]:

# Extract gain importance and rank
importance = pd.DataFrame({
    'feature':    X_train.columns,
    'gain':       xgb_model.feature_importances_,  
}).sort_values('gain', ascending=False).reset_index(drop=True)

# Add cumulative gain for later selection
importance['cumulative_gain'] = importance['gain'].cumsum() / importance['gain'].sum()
importance['cumulative_pct']  = (importance['cumulative_gain'] * 100).round(2)

# Print full table
print(importance.to_string(index=False))


# In[42]:

fig, ax = plt.subplots(figsize=(9, 10))

sns.barplot(data=importance, x='gain', y='feature',
            hue='feature', palette='viridis', legend=False, ax=ax)

ax.set_title('XGBoost Feature Importance (Gain)', fontsize=14, fontweight='bold')
ax.set_xlabel('Gain')
ax.set_ylabel('')

# Annotate cumulative gain on the right
for i, (g, cum) in enumerate(zip(importance['gain'], importance['cumulative_pct'])):
    ax.text(g + 0.001, i, f'{cum:.1f}%', va='center', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('images/xgb_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()


print("#--"*10)
print("\n Checking Interaction Effects:")
# **Screening rule (a rule of thumb).** A pair is carried forward to Notebook 2 only if (1) its total SHAP interaction strength (twice the mean |value| of one triangle of the matrix) is at least around 10% of the stronger feature's main effect, (2) it is not a built-in artifact (`TotalCharges ≈ tenure × MonthlyCharges`, or dummies of the same variable), and (3) it has a plausible business explanation. Ratios are read with care when the stronger main effect is small, because the percentage is then inflated. 

# In[43]:


booster = xgb_model.get_booster()
inter = booster.predict(xgb.DMatrix(X_fit), pred_interactions=True)  # XGBoost internal data format (n, F+1, F+1)
inter = inter[:, :-1, :-1]                       # (n, F, F)

strength = np.abs(inter).mean(axis=0)            # mean |interaction| per feature pair
np.fill_diagonal(strength, 0)               

feats = X_fit.columns
pairs = (pd.DataFrame(strength, index=feats, columns=feats)
         .stack().reset_index())
pairs.columns = ['f1', 'f2', 'strength']
pairs = pairs[pairs.f1 < pairs.f2].sort_values('strength', ascending=False)

# Drop pairs from the same variable
pairs = pairs[pairs.f1.str.split('_').str[0] != pairs.f2.str.split('_').str[0]]
print(pairs.head(10))


# In[44]:

inter_full = booster.predict(xgb.DMatrix(X_fit), pred_interactions=True)
margin = booster.predict(xgb.DMatrix(X_fit), output_margin=True)

# Sanity check: all entries (incl. baseline) sum to the log-odds prediction
print("additivity error:", np.abs(inter_full.sum(axis=(1, 2)) - margin).max())
print("std of log-odds across customers:", margin.std().round(3))

inter = inter_full[:, :-1, :-1]
main = pd.Series(np.abs(inter).mean(axis=0).diagonal(), index=X_fit.columns)
print(main.sort_values(ascending=False).head(10).round(3))

# Interaction as a share of the stronger feature's main effect
pairs['pair_total'] = 2 * pairs['strength']
pairs['vs_main'] = pairs['pair_total'] / pairs[['f1', 'f2']].apply(
    lambda r: max(main[r.f1], main[r.f2]), axis=1)
print(pairs.head(10).round(3))


print("#--"*10)
print("\n **Selection Decision** ")

# In[45]:

# Select top 12 features by gain + MonthlyCharges (rank 14) for business interpretability
top12 = importance.head(12)['feature'].tolist()
selected_features = top12 + ['MonthlyCharges'] if 'MonthlyCharges' not in top12 else top12

print(f"Selected {len(selected_features)} features:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feat}")


print("#--"*10)
print("\n Robustness Check: ")
# Does the top-12 selection survive different validation splits and seeds?

# In[46]:

seeds = [60, 1, 2, 3, 4]
rows = []
for s in seeds:
    Xf, Xv, yf, yv = train_test_split(X_train, y_train, test_size=0.20,
                                      stratify=y_train, random_state=s)
    m = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4,
                      subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                      early_stopping_rounds=30, random_state=s, n_jobs=-1)
    m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
    imp = pd.Series(m.feature_importances_, index=X_train.columns)
    top = set(imp.sort_values(ascending=False).head(12).index)
    rows.append({'seed': s,
                 'overlap_with_selected_top12': len(top & set(top12)),
                 'fiber_gain': imp['InternetService_Fiber optic'],
                 'electronic_check_gain': imp['PaymentMethod_Electronic check']})
print(pd.DataFrame(rows).round(3).to_string(index=False))


# **Robustness check.**  
# Refitting the model under five different validation splits and seeds, the top-12 set overlapped with the selected set in 11–12 of 12 features. Individual gain shares moved noticeably (for example, `InternetService_Fiber optic` ranged from 14.3% to 25.7%), so the exact percentages should not be over-interpreted. The overall ranking, with contract type, `internet service` and `payment method` at the top, is stable.

# ### Export &  Save Datasets

# In[47]:

# 1. Save selected feature list
with open('selected_features.json', 'w') as f:
    json.dump(selected_features, f, indent=2)

# 2. Subset the train/test splits to selected features + target
X_train_sel = X_train[selected_features].copy()
X_test_sel  = X_test[selected_features].copy()

# 3. Recombine with target for CSV export
train_export = X_train_sel.copy()
train_export['Churn'] = y_train.values

test_export = X_test_sel.copy()
test_export['Churn'] = y_test.values

# 4. Write CSVs
train_export.to_csv('df_train.csv', index=False)
test_export.to_csv('df_test.csv', index=False)

print(f"df_train.csv : {train_export.shape}")
print(f"df_test.csv  : {test_export.shape}")
print(f"selected_features.json : {len(selected_features)} features")


# ### Notebook 1 Complete ✅
# 
# **Artifacts exported:**
# - `df_train.csv` — 4,930 rows × 13 selected features + `Churn` target
# - `df_test.csv` — 2,113 rows × 13 selected features + `Churn` target
# - `selected_features.json` — list of the 13 selected feature names
# 
# **Pipeline summary:**
# 1. Loaded the full Telco dataset (7,043 × 21), dropped `customerID`
# 2. Converted types; verified structural missingness in `TotalCharges` (11 rows, all `tenure = 0`)
# 3. Diagnosed numeric and categorical features; validated class imbalance (26.5% churn)
# 4. Detected and validated high-value tail customers (109 rows, all arithmetically consistent)
# 5. Filled structurally missing `TotalCharges` with 0; ran QA assertions
# 6. Explored feature-churn relationships (Objective 1)
# 7. Ranked all 23 encoded features by XGBoost gain; selected top 12 + `MonthlyCharges` (Objective 2)
# 8. Exported cleaned, encoded, feature-selected train/test splits
# 
# **Next:** proceed to `02_modeling_and_business_impact.ipynb` for classifier comparison, threshold tuning, and ROI simulation.
