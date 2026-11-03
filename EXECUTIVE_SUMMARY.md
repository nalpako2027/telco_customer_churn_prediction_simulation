# Executive Summary: Customer Churn Prediction and Retention Strategy Analysis with Business Impact Simulation

**Author:** Orhan Kaplan · **Data:** IBM Telco Customer Churn dataset (7,043 customers) · **Full technical detail:** see the [README](README.md)

---

## The Business Problem
Customer churn is a direct revenue leak. In this dataset, **26.5% of customers (1,869 of 7,043) left**, representing roughly **\$139K in lost monthly recurring revenue**. Retention budgets are limited, so the question is not only *who* will leave, but *who is worth contacting, and how much can we afford to spend on them?*

## What Was Done
1. **Diagnosed** which customer attributes are associated with churn.
2. **Selected** the most informative of 23 encoded features (XGBoost ranking) to keep the final model simple and explainable.
3. **Built and compared** two prediction models (Logistic Regression and k-Nearest Neighbors) on customers the models had never seen.
4. **Grouped customers** into four risk tiers (Critical, High, Medium, Low) based on predicted churn probability.
5. **Simulated the 12-month financial return (ROI)** of retention offers (hypothetical interventions) at different price points and success rates.

## Key Findings

**1. Churn is concentrated in a recognizable profile.**
Contract type is the strongest signal: month-to-month customers churn at ~43% versus ~3% for two-year contracts. Fiber-optic customers churn at ~42% (DSL ~19%), and electronic-check payers at ~45% (about 15–17% for automatic payment). The highest-risk profile is a **month-to-month, fiber-optic customer paying by electronic check**. Churners also tend to be newer (18 months average tenure vs. 38 months for retained customers).

**2. Churn can be predicted reliably enough to act on.**
Logistic Regression was the better model: it **identifies 77.2% of customers who actually churn** (ROC-AUC 0.85), and its probabilities are well calibrated, so they can be used directly in financial planning.

**3. A small group holds most of the opportunity.**
The Critical and High tiers contain **464 customers (22% of the base)** but account for **56.6% of expected monthly revenue at risk** and **55% of all churners**. Observed churn runs from **80.9% in the Critical tier to 7.3% in the Low tier**, so the tiers separate risk meaningfully.

| Risk tier | Customers | Avg. monthly bill | Observed churn |
| :--- | :---: | :---: | :---: |
| Critical | 47 | USD 84.03 | 80.9% |
| High | 417 | USD 77.77 | 64.7% |
| Medium | 471 | USD 68.86 | 35.5% |
| Low | 1,178 | USD 55.89 | 7.3% |

**4. Offer cost decides whether a campaign pays off.**
Assuming a retention offer saves 20% of the targeted churners, the 12-month return on the 464 targeted customers is:

| Monthly offer cost | Net benefit | ROI |
| :---: | :---: | :---: |
| USD 2 | USD 48,050 | 431.5% |
| USD 5 | USD 31,346 | 112.6% |
| USD 7 | USD 20,210 | 51.9% |
| USD 10 | USD 3,506 | 6.3% |
| USD 15 | USD −24,334 | −29.1% |

The break-even offer is about **USD 10.63 per month**. Low-cost perks (e.g., service upgrades or bundles) are far more attractive than deep price cuts.

## Recommendations
1. **Prioritize the Critical and High tiers** for outreach and leave the Low tier untouched to protect margin.
2. **Favor low-cost, non-price incentives** (roughly \$7 per month or less in effective cost) over deep discounts.
3. **Pilot before scaling:** run a controlled A/B test (offer vs. no-offer group, several offer levels) to measure the real save rate before committing a full budget.
4. **Review the conditions behind the risk profile** (month-to-month contracts, fiber-optic service quality, manual payment methods) as areas to investigate, not as proven causes (see below).

## Important Caveats
- **Associations, not causation.** The data show which customers are likely to leave, not what would make them stay. The dataset contains no record of past retention offers.
- **The 20% save rate is an assumption**, not a measured result, and it is applied equally to every offer level. All ROI figures are therefore **upper-bound scenarios** that count revenue (not margin) and assume saved customers stay the full 12 months.
- **Targeting by risk is not targeting by responsiveness.** The highest-risk customers are not necessarily the most persuadable, and some offers can backfire.
- Financial figures refer to the **2,113-customer test sample**, not the full customer base, and should be scaled accordingly.

## Next Steps
Run the A/B field test, use historical offer data (if available) for quasi-experimental estimates, build uplift models to find who actually responds to offers, and monitor model performance as pricing and service tiers evolve.
