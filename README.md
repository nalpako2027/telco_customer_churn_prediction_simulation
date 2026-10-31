![License](https://img.shields.io/github/license/nalpako2027/telco_customer_churn_prediction_simulation?style=for-the-badge)
![Last Commit](https://img.shields.io/github/last-commit/nalpako2027/telco_customer_churn_prediction_simulation?style=for-the-badge)
![Python](https://img.shields.io/badge/python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/orhan-kaplan-phd-5a4a84212/)

# ✨ Telco Customer Churn Analysis, Feature Selection & Predictive Modeling Pipeline with Retention ROI Simulation

## 📝 1. Summary
This repository contains an end-to-end data science and business analytics pipeline on the **IBM Telco Customer Churn dataset**: exploratory analysis, XGBoost-based feature selection, Logistic Regression vs. k-Nearest Neighbors (KNN) modeling, customer risk tranches, and a financial ROI simulation of retention offers. It exists to identify high-risk churn segments and to estimate, under explicit assumptions, what proactive retention could be worth.

## 🎯 2. Objectives
**Business case.** Customer attrition is a major revenue leak in telecommunications: 1,869 of 7,043 customers (26.5%) churned, roughly USD 139K in monthly recurring revenue at an average churner bill of about USD 74. This project shifts the strategy from reactive customer care to proactive, targeted retention.

**Technical goals.**
1. Describe the relationship between customer attributes (tenure, contract, services, billing) and churn.
2. Rank all predictors by XGBoost gain importance and select a compact feature set.
3. Predict churn from the selected features and compare Logistic Regression with KNN under class imbalance.
4. Convert predicted probabilities into prioritized risk tranches (Critical / High / Medium / Low).
5. Simulate the 12-month financial impact (cost vs. protected revenue) of targeting the high-risk tranches.

## 📂 3. Dataset
- **Source:** IBM Telco Customer Churn dataset (`Telco-Customer-Churn.csv`), https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data (Apache 2.0 License).
- **Size:** 7,043 customers × 21 columns (20 after dropping `customerID`); 5,174 stayed (73.5%), 1,869 churned (26.5%).
- **Preparation before modeling:** `customerID` dropped; `TotalCharges` converted to numeric; 11 structurally missing `TotalCharges` values (all `tenure = 0`, all non-churners) filled with 0; redundant levels ("No internet service", "No phone service") collapsed into "No". No rows were removed. A stratified 70/30 split (`random_state=60`) gives 4,930 training and 2,113 test rows.

## 🔧 4. Methodology
**Notebook 1: `01_eda_and_feature_selection.ipynb`**
1. **Cleaning & QA:** type conversion, programmatic `assert` checks (no negative charges or tenure; customers with `tenure == 0` carry zero `TotalCharges`; redundant-level checks), and an IQR outlier check (no outliers found, none removed).
2. **EDA:** churn rate by category, churned vs. retained means, Pearson correlations (numeric), Cramér's V (categorical).
3. **Encoding & split:** one-hot encoding (23 predictors), stratified 70/30 split.
4. **Feature selection:** XGBoost used only as a ranking tool (`max_depth=4`, `learning_rate=0.05`, early stopping on a validation split of the training set). Top 12 features by gain (84.8% cumulative gain) plus `MonthlyCharges` (force-included as the business-critical revenue variable) = **13 features**. A 5-seed robustness check kept 11–12 of the top 12, and a SHAP interaction screen found only modest interactions, so no interaction terms were added.

**Notebook 2: `02_modeling_and_business_impact.ipynb`**
1. **Models:** MinMax-scaled features; KNN (`k=41`, distance-weighted; chosen by 5-fold stratified CV on PR-AUC) and Logistic Regression (`C=10`, tuned by CV, no class weights).
2. **Imbalance handling:** through the decision threshold rather than reweighting: thresholds maximize F1 on out-of-fold training predictions (KNN 0.33, LR 0.31), so predicted probabilities stay calibrated for the ROI simulation.
3. **Diagnostics:** VIF, spline likelihood-ratio tests for logit linearity, calibration (Brier score, slope), ROC and PR curves, paired bootstrap and McNemar tests.
4. **Risk tranches & ROI:** probability cut-offs at 0.25 / 0.50 / 0.75; Critical + High tranches targeted; offers of USD 2–15 per month over 12 months; 20% base save rate with a 15–50% sensitivity grid.

## 📊 5. Key Findings & Model Performance

### Test-set comparison (each model at its own F1-optimal threshold)

| Metric | KNN | Logistic Regression |
| :--- | :---: | :---: |
| Accuracy | 75.7% | **76.7%** |
| Recall | 72.4% | **77.2%** |
| Specificity | **76.9%** | 76.5% |
| Precision | 53.1% | **54.3%** |
| F1-Score | 61.3% | **63.8%** |
| ROC-AUC | 0.815 | **0.852** |
| PR-AUC | 0.601 | **0.657** |

**In practice:** Logistic Regression is the better model to deploy. It flags 77.2% of actual churners, and its ROC-AUC and PR-AUC advantages are statistically supported (paired bootstrap: ROC-AUC +0.037, 95% CI +0.024 to +0.051; PR-AUC +0.056, 95% CI +0.018 to +0.093). The gap at the classification level is small (McNemar p = 0.20). Accuracy is a weak yardstick here, since "everyone stays" already scores 73.5%. Probabilities are well calibrated (Brier score 0.1331, calibration slope 1.07), which makes them usable for the financial simulation.

**Strongest predictors:** XGBoost gain ranks `Contract_Two year` (21.2%), `InternetService_Fiber optic` (20.8%), `Contract_One year` (12.4%), `InternetService_No` (6.1%) and `PaymentMethod_Electronic check` (4.9%) at the top; the top 5 carry 65.3% of total gain. In Logistic Regression (odds ratios, holding other features fixed): two-year vs. month-to-month contract 0.20 (95% CI 0.13–0.30), one-year 0.54 (0.42–0.69), fiber optic vs. DSL 3.3 (2.2–4.9), no internet service vs. DSL 0.27 (0.18–0.41). The highest-risk profile is a month-to-month, fiber-optic customer paying by electronic check.

## 🧩 6. Business Questions & Key Insights

**Q1. How do customer attributes relate to churn?** Month-to-month customers churn at ~43% vs. ~3% for two-year contracts; fiber optic ~42% vs. DSL ~19%; electronic check ~45% vs. ~15–17% for automatic payment. Churners average 18.0 months of tenure vs. 37.6 for retained customers and pay USD 74.44/month vs. 61.27. `gender` shows no separation.

**Q2. Which features carry the strongest signal?** Contract type, internet service type, payment method and tenure (see Section 5). `tenure`, `MonthlyCharges` and `TotalCharges` are strongly overlapping (`TotalCharges` ≈ `tenure` × `MonthlyCharges`; VIF up to 21.7), so their coefficients are not interpreted individually.

**Q3. Can churn be predicted, and which model is better?** Yes. Logistic Regression outperforms KNN on every headline metric except specificity (Section 5).

**Q4. Who should be targeted?** Test-set risk tranches:

| Risk tranche | Customers | Avg. monthly bill | Observed churn rate |
| :--- | :---: | :---: | :---: |
| Critical (P ≥ 0.75) | 47 | USD 84.03 | 80.9% |
| High (0.50–0.75) | 417 | USD 77.77 | 64.7% |
| Medium (0.25–0.50) | 471 | USD 68.86 | 35.5% |
| Low (< 0.25) | 1,178 | USD 55.89 | 7.3% |

Critical + High cover 464 customers (22.0% of the test base) and 56.6% of expected monthly revenue at risk (USD 23,220.95 of USD 41,016.44), and reach 54.9% of all test-set churners.

**Q5. What is the financial impact?** With 464 targeted customers and a 20% save rate (62 churners saved), gross revenue protected over 12 months is USD 59,186.41. ROI depends strongly on offer cost:

| Offer (USD/month) | Campaign cost | Net benefit | ROI |
| :---: | :---: | :---: | :---: |
| 2 | USD 11,136 | USD 48,050 | 431.5% |
| 5 | USD 27,840 | USD 31,346 | 112.6% |
| 7 | USD 38,976 | USD 20,210 | 51.9% |
| 10 | USD 55,680 | USD 3,506 | 6.3% |
| 15 | USD 83,520 | USD −24,334 | −29.1% |

Break-even sits at roughly USD 10.63/month at a 20% save rate. The break-even save rate rises with offer size, from about 4% at USD 2 to about 28% at USD 15. At USD 7, ROI ranges from +12.7% (15% save rate) to +277.2% (50%). These are upper-bound scenarios (see Section 10).

**Q6. [UPDATE]** Do the `tenure × Contract_Two year` and `tenure × InternetService_Fiber optic` interactions improve Logistic Regression? Flagged as plausible in Notebook 1; result to be added.

## ⚙️ 7. Requirements
- Python [UPDATE: version]
- pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, xgboost, shap, statsmodels, prettytable [UPDATE: version numbers]
- Jupyter Notebook

## ▶️ 8. How to Reproduce
1. Clone the repository and install the dependencies:
   ```bash
   git clone https://github.com/nalpako2027/telco_customer_churn_prediction_simulation.git
   cd telco_customer_churn_prediction_simulation
   pip install -r requirements.txt
   ```
2. Download `Telco-Customer-Churn.csv` from the IBM link in Section 3 and place it in the repository root (create an `images/` folder if it does not exist; the notebooks save figures there).
3. Run `01_eda_and_feature_selection.ipynb` first. It exports `df_train.csv`, `df_test.csv` and `selected_features.json`.
4. Run `02_modeling_and_business_impact.ipynb`, which reads those three files.

## 🗂️ 9. Project Structure
```
├── 01_eda_and_feature_selection.ipynb      # Cleaning, QA, EDA, XGBoost feature selection
├── 02_modeling_and_business_impact.ipynb   # KNN vs. Logistic Regression, risk tranches, ROI simulation
├── Telco-Customer-Churn.csv                # Raw IBM data (download separately)
├── df_train.csv / df_test.csv              # Train/test splits with the 13 selected features (generated by notebook 1)
├── selected_features.json                  # List of the 13 selected features (generated by notebook 1)
├── images/                                 # Figures saved by the notebooks
├── requirements.txt                        # Python dependencies
└── README.md
```

## ⚠️ 10. Limitations & Further Recommendations
**Limitations**
- **Association, not causation.** The models establish predictive associations between customer attributes and churn. The dataset contains no record of past retention offers, so the effect of a discount cannot be identified from it.
- **ROI figures are upper-bound scenarios.** The 20% save rate is an assumption, applied flatly to every offer level (the closest field evidence suggests small discounts may work as well as large ones). Every saved customer is assumed to stay 12 months, revenue (not margin) is counted, and the discount cost is applied to all 464 targeted customers.
- **Heterogeneous response.** High-risk customers are not necessarily the most persuadable, and some proactive offers can increase churn (Ascarza, 2018; Ascarza et al., 2016).
- **Collinearity and non-linearity.** `tenure`, `MonthlyCharges` and `TotalCharges` overlap strongly, and linearity of the logit is rejected for `tenure` and `TotalCharges`. A spline model improves PR-AUC by only about 0.014, so the linear model is kept for interpretability.
- **Feature selection.** XGBoost gain splits credit among correlated features, so a low rank does not mean a weak predictor. XGBoost itself was not tuned or evaluated.
- **Untested explanations.** The reasons KNN underperforms and the influence of outliers (e.g., Cook's distance) were not tested directly.

**Next steps**
1. Run a randomized A/B field test with a no-offer control and several discount levels to estimate the causal effect and the offer → save-rate curve.
2. Where historical offer data exists, use quasi-experimental designs (propensity score matching, regression discontinuity, difference-in-differences).
3. Fit uplift or causal forest models to find which customers actually respond to offers.
4. Monitor feature and performance drift as pricing and internet tiers evolve.

## 📚 11. References
- IBM. Telco Customer Churn dataset. https://github.com/IBM/telco-customer-churn-on-icp4d
- Ascarza, E. (2018). Retention futility: Targeting high-risk customers might be ineffective. *Journal of Marketing Research, 55*(1), 80–98.
- Ascarza, E., Iyengar, R., & Schleicher, M. (2016). The perils of proactive churn prevention using plan recommendations: Evidence from a field experiment. *Journal of Marketing Research, 53*(1), 46–60.
- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An introduction to statistical learning: With applications in Python.* https://www.statlearning.com/
- Lomax, R. G., & Hahs-Vaughn, D. L. (2012). *An Introduction to Statistical Concepts* (3rd ed.). Routledge.
- Van den Goorbergh, R., Van Smeden, M., Timmerman, D., & Van Calster, B. (2022). The harm of class imbalance corrections for risk prediction models. *Journal of the American Medical Informatics Association, 29*(9), 1525–1534.
- Yang, J., Eckles, D., Dhillon, P., & Aral, S. (2022). Targeting for long-term outcomes (arXiv:2010.15835v2). https://arxiv.org/abs/2010.15835

## 12. License
See the `LICENSE` file in this repository [UPDATE: state the license type]. The raw data comes from IBM under the Apache 2.0 License and was modified for type alignment, validation checks and train/test splits.

## 👤 13. Author / Attribution
**Orhan Kaplan** · [LinkedIn](https://www.linkedin.com/in/orhan-kaplan-phd-5a4a84212/) · [UPDATE: portfolio link]

If you use this project in research, publications, presentations, or other work, please cite or acknowledge:

Orhan Kaplan  
Telco Customer Churn Analysis, Feature Selection & Predictive Modeling Pipeline with Retention ROI Simulation  
GitHub URL: https://github.com/nalpako2027/telco_customer_churn_prediction_simulation.git
