# Decision Log
---

## DL-001 — Dataset Selection
**Date:** 2026-03-12
**Decision:** ACS PUMS 2023 (U.S. Census Bureau) selected as primary dataset
**Rationale:** Official government microdata, annually updated, 
~1.5M national records. Reflects current labor market conditions 
unlike historical alternatives.
**Verified:** Virginia pull (FIPS 51) returned 88,928 records at 647KB parquet

---

## DL-002 — Income Threshold
**Date:** 2026-03-12
**Decision:** Binary target threshold set at $75,000 annual wage income
**Rationale:** Approximates 2023 US median household income. 
Reflects current financial risk thresholds used in federal 
program eligibility determinations.

---

## DL-003 — Development Scope
**Date:** 2026-03-12
**Decision:** Virginia (FIPS 51) used for pipeline development and testing
**Rationale:** 88,928 records sufficient for model development with 
significantly faster iteration cycles than national pull (~1.5M records).
National pull reserved for final model training.

---

## DL-004 — person_weight Handling
**Date:** 2026-03-12
**Decision:** person_weight excluded from OLS and Ridge inputs.
Used as sample_weight in XGBoost training only.
**Rationale:** Census sampling weight is a survey methodology artifact,
not a personal characteristic. Linear baseline models are kept clean
for interpretability. XGBoost uses it to produce population-representative
predictions aligned with federal agency requirements.

---

## DL-005 — Baseline Model Results
**Date:** 2026-03-12
**Model:** Logistic Regression
**AUC-ROC:** 0.9108 | **F1:** 0.6508 | **Precision:** 0.6856 | **Recall:** 0.6195
**Notes:** Strong AUC for a baseline. F1 reflects 78/22 class imbalance.
hours_per_week is the dominant predictor (coef 1.72). Ridge and XGBoost
must demonstrate meaningful improvement to justify added complexity.

---

## DL-006 — Ridge Model Results
**Date:** 2026-03-12
**Model:** Ridge Logistic Regression (LogisticRegressionCV)
**AUC-ROC:** 0.9108 | **F1:** 0.6507 | **Best C:** 100.0
**Delta vs Baseline:** AUC 0.0000 | F1 -0.0001
**Finding:** CV selected maximum C value (weakest regularization),
producing results identical to baseline. Features are sufficiently
independent — L2 regularization provides no benefit on this dataset.
Linear model ceiling reached at AUC 0.91. Non-linear relationships
in occupation and class_of_worker are the primary motivation for
XGBoost as the next stage.

---

## DL-007 — person_weight excluded from StandardScaler
**Date:** 2026-03-12
**Decision:** person_weight removed from NUMERIC_FEATURES in config.py
**Rationale:** Census sampling weight is a raw population count. Scaling
to mean=0 destroys its meaning and produces negative values rejected
by XGBoost. Passed as raw values to sample_weight parameter.

---

## DL-008 — XGBoost selected as production model
**Date:** 2026-03-12
**Model:** XGBoost + Optuna (30 trials)
**AUC-ROC:** 0.9506 | **F1:** 0.7633
**Delta vs baseline:** AUC +0.0398 | F1 +0.1125
**Rationale:** Meaningful improvement on both primary metrics justifies
added complexity. occupation and class_of_worker ranked 2nd and 4th in
feature importance — confirming non-linear signal in those features
that linear models could not capture. scale_pos_weight: 2.43 found by
Optuna significantly improved minority class recall (0.62 → 0.85).