# Telco Customer Churn Analysis & Predictive Modeling Pipeline

This repository contains an end-to-end data science and business analytics pipeline designed to identify high-risk customer churn segments and evaluate the financial impact of proactive marketing interventions using the **IBM Telco Customer Churn dataset**. Features include custom logic assertions, optimization of decision thresholds, multi-model ROC evaluation, and simulation of the financial ROI of customer retention strategies.

---

## 🎯 Project Overview & Objectives
Customer attrition (churn) is a major revenue leak in the telecommunications industry. This project shifts the business strategy from reactive customer care to proactive retention. 

**Key Objectives:**
1. **Data Management & QA:** Establish a robust cleaning pipeline to handle data type anomalies and build strict logic validation constraints.
2. **Exploratory Data Analysis (EDA):** Diagnose overall distributions and behavior trends of continuous financial features (`tenure`, `MonthlyCharges`, `TotalCharges`).
3. **Predictive Modeling:** Train and optimize **K-Nearest Neighbors (KNN)** and **Logistic Regression** classifiers to handle severe class imbalance.
4. **Financial Impact Simulation:** Build a financial ROI framework to simulate intervention costs versus protected lifetime revenue.

---

## 🏗️ Jupyter Notebook Architecture

The code repository is organized into a highly structured, business-ready pipeline:
1. **Executive Summary & Strategic Objectives** (Problem mapping & research framing)
2. **Data Management & Preprocessing Pipeline** (`pd.to_numeric` coercion & missing value isolation)
3. **Data Quality Assurance (QA) & Logic Assertion Checks** (Programmatic error-catching)
4. **Exploratory Data Analysis (EDA)** (Baseline frequency and univariate feature distributions)
5. **Comparative Analysis & Key Business Insights** (Group means & boxplot outlier validation)
6. **Predictive Modeling & Evaluation Strategy** (Feature scaling, custom thresholding, ROC/AUC plots)
7. **Data-Driven Strategic Recommendations & ROI Simulation** (Financial optimization models)

---

## 🛠️ Data Quality Assurance & Logic Validations
Real-world behavioral datasets often contain structural quirks. This pipeline implements custom programmatic `assert` statements to ensure data integrity before modeling:

* **New Customer Constraint:** Programmatically verifies that new accounts (`tenure == 0`) accurately carry zero or null balances in `TotalCharges`.
* **The Lifetime Bill Ceiling:** Implements a strict business logic check to catch database corruption:
  $$\text{TotalCharges} \leq (\text{tenure} \times \text{MonthlyCharges} \times 1.5)$$
  *Note: Low-tenure records (tenure $\leq$ 5 months) are safely bypassed from the ceiling check to accommodate baseline upfront hardware activation fees or early account plan modifications.*

---

## 📈 Model Performance & Diagnostic Evaluation

The original dataset features a distinct **73.5% No Churn / 26.5% Churn class imbalance**. Evaluating classifiers strictly on raw accuracy leads to a false sense of security. To solve this, models were heavily tuned:
* **KNN Classifier:** Tailored using optimal hyperparameter tuning ($k=29$) and balanced via distance-weighted spatial voting (`weights='distance'`).
* **Logistic Regression:** Regularized ($C=0.1$, `max_iter=10000`) and balanced using automated mathematical class-weighting (`class_weight='balanced'`).

### Final Comparison Matrix (Test Data Performance)

| Metric | kNN Model | Logistic Regression Model | Evaluation Summary |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 77.9% | 73.4% | kNN leans heavily on the easier-to-predict majority class. |
| **Recall** | 44.9% | **75.1%** | **Logistic Regression catches 30.2% more actual churners.** |
| **Specificity** | 90.2% | 72.8% | kNN is more conservative regarding active customers. |
| **Precision** | 63.1% | 50.8% | Logistic Regression accepts false alarms to minimize revenue leaks. |
| **F1-Score** | 52.4% | **60.6%** | **Logistic Regression is the mathematically superior model.** |

### Multi-Model Threshold Diagnostics (ROC Curve)
By computing unique predicted probabilities and running an optimized custom threshold pipeline, the **Balanced Logistic Regression model emerged as the definitive champion with an excellent AUC score of 0.813**, outperforming the KNN Classifier (AUC = 0.784).

---

## 💰 Business Impact & Financial ROI Simulation

To demonstrate actionable value to stakeholders, a Financial Return on Investment (ROI) matrix was simulated on the highest risk tiers (851 targeted accounts representing **USD 64,960.80** in immediate monthly recurring revenue). 

The analysis reveals a critical operational threshold regarding campaign design:

### Strategy 1: Direct Cash Discount Campaign ($15/month credit)
* **The Math:** Offering a USD 15.00/month price cut to all 851 targeted accounts across a 12-month contract lifespan.
* **The Outcome:** Successfully saves 172 true churners and protects **USD 157,554.75** in gross 12-month revenue. However, the overhead cost of the 420 false alarms compresses the campaign's net profit to **USD 4,374.75**, resulting in a marginal **2.9% ROI**.

### Strategy 2: Low-Marginal-Cost Incentive Campaign ($2/month internal cost)
* **The Math:** Pivoting the campaign to low-cost, high-perceived-value perks (e.g., automated router hardware upgrades or free premium streaming data bundles).
* **The Outcome:** Keeps the identical 172 customer volume saved and gross revenue protected, but slashes 12-month campaign overhead down to **USD 20,424.00**. 
* **The Financial Win:** Net benefit explodes to **USD 137,130.75**, yielding an exceptional **671.4% ROI** and cleanly proving the business value of the data science framework.


### Strategy 3: Medium Cost Discount Campaign ($7/month credit)
* **The Math:** Offering a USD 7.00/month price cut to all targeted accounts across a 12-month contract lifespan.
* **The Outcome:** Successfully saves 172 true churners and protects **USD 86,070.75** in gross 12 month revenue, resulting in a large **120.4% ROI**.

---
### Methodological Limitation:   
The models establish predictive correlations between billing metrics and customer behavior. The analysis identifies strong statistical associations and predictive patterns within the data, but causality cannot be claimed. Therefore, all intervention simulations are based on directional business assumptions rather than experimental causal impact data.

---
## 🚀 How to Run the Project
1. Clone this repository to your machine.
2. Ensure you have the required libraries installed (bash: Instantly replicate the environment):
   pip install -r requirements.txt
3. Open your terminal, boot up your interface, and run notebook chunks sequentially:
   ```bash
   jupyter notebook
   ```

