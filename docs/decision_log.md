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

DL-008 — XGBoost Selected as Production Model
Date: 2026-03-12
Model: XGBoost + Optuna (30 trials)
AUC-ROC: 0.9506 | F1: 0.7633 | Best CV AUC: 0.9484
Delta vs Baseline:

AUC: +0.0398 (+4.4%)
F1:  +0.1125 (+17.3%)
Recall: +0.2351 — model captures significantly more high-income individuals

Best Parameters (Optuna):

n_estimators: 403
max_depth: 5
learning_rate: 0.0432
scale_pos_weight: 2.43
gamma: 0.438

Rationale: Meaningful improvement on both primary metrics justifies
added complexity. occupation and class_of_worker ranked 2nd and 4th in
feature importance — confirming non-linear signal those features carry
that logistic regression could not capture. scale_pos_weight: 2.43 found
by Optuna significantly improved minority class recall (0.62 → 0.85).

DL-009 — Fairness Audit Results
Date: 2026-03-12
Gate: PASSED — all groups within ±0.20 PPR threshold
Overall PPR: 0.2686
Per-Group Positive Prediction Rates:
GroupNPPRDeltaAUCWhite alone9,6850.27870.01010.9539Black or African American alone2,0860.17070.09790.9196American Indian alone680.35290.08430.9086Asian alone1,0590.41270.14410.9522Some other race alone4480.16740.10120.9414Two or more races1,0440.26340.00520.9480Male6,9290.32110.05250.9456Female7,4780.21980.04880.9543Native born12,4450.25990.00870.9507Foreign born1,9620.32360.05500.9488
Findings:

Largest inter-group disparity — Race:
Black or African American (PPR 0.17) vs Asian (PPR 0.41) — a 0.24
absolute gap. Each group individually passes the gate measured against
overall PPR, but the inter-group gap warrants monitoring in production.
Likely reflects underlying income distribution in Virginia 2023 ACS
data rather than model-introduced bias — requires further investigation
before deployment in high-stakes contexts.
Sex disparity:
Male PPR (0.32) vs Female PPR (0.22) — 10 point gap. Both groups
show strong and nearly equal AUC (0.9456 / 0.9543), indicating the
model predicts well for both. Disparity reflects income distribution
in the underlying data.
American Indian alone — small sample warning:
Only 68 records in the Virginia test set. Precision 0.375 with
recall 0.90 indicates the model over-predicts high income for this
group. AUC 0.9086 is the lowest across all race groups. Metrics
unreliable at this sample size — national data pull will produce
more representative results for this group.
Nativity:
Foreign born PPR (0.32) slightly higher than native born (0.26).
AUC nearly identical (0.9488 / 0.9507) — no performance degradation
across nativity groups.

Risk Response (NIST AI RMF MANAGE 2.2):
Gate passed for Virginia development data. The following monitoring
requirements are established for production deployment:

Re-run fairness audit after national data pull (STATE_CODE="*")
Monitor per-group PPR drift monthly via Evidently AI
Flag any group exceeding ±0.15 PPR delta (tighter than gate threshold)
for human review before next retraining cycle
Document American Indian group metrics after national pull## DL-007 — person_weight Excluded from StandardScaler
**Date:** 2026-03-12
**Decision:** person_weight removed from NUMERIC_FEATURES in config.py
**Rationale:** Census sampling weight is a raw population count.
Scaling to mean=0 destroys its meaning and produces negative values
rejected by XGBoost. Passed as raw values to sample_weight parameter.

---

## DL-008 — XGBoost Selected as Production Model
**Date:** 2026-03-12
**Model:** XGBoost + Optuna (30 trials)
**AUC-ROC:** 0.9506 | **F1:** 0.7633 | **Best CV AUC:** 0.9484

**Delta vs Baseline:**
- AUC: +0.0398 (+4.4%)
- F1:  +0.1125 (+17.3%)
- Recall: +0.2351 — model captures significantly more high-income individuals

**Best Parameters (Optuna):**
- n_estimators: 403
- max_depth: 5
- learning_rate: 0.0432
- scale_pos_weight: 2.43
- gamma: 0.438

**Rationale:** Meaningful improvement on both primary metrics justifies
added complexity. occupation and class_of_worker ranked 2nd and 4th in
feature importance — confirming non-linear signal those features carry
that logistic regression could not capture. scale_pos_weight: 2.43 found
by Optuna significantly improved minority class recall (0.62 → 0.85).

---

## DL-009 — Fairness Audit Results
**Date:** 2026-03-12
**Gate:** PASSED — all groups within ±0.20 PPR threshold
**Overall PPR:** 0.2686

**Per-Group Positive Prediction Rates:**

| Group | N | PPR | Delta | AUC |
|---|---|---|---|---|
| White alone | 9,685 | 0.2787 | 0.0101 | 0.9539 |
| Black or African American alone | 2,086 | 0.1707 | 0.0979 | 0.9196 |
| American Indian alone | 68 | 0.3529 | 0.0843 | 0.9086 |
| Asian alone | 1,059 | 0.4127 | 0.1441 | 0.9522 |
| Some other race alone | 448 | 0.1674 | 0.1012 | 0.9414 |
| Two or more races | 1,044 | 0.2634 | 0.0052 | 0.9480 |
| Male | 6,929 | 0.3211 | 0.0525 | 0.9456 |
| Female | 7,478 | 0.2198 | 0.0488 | 0.9543 |
| Native born | 12,445 | 0.2599 | 0.0087 | 0.9507 |
| Foreign born | 1,962 | 0.3236 | 0.0550 | 0.9488 |

**Findings:**

1. **Largest inter-group disparity — Race:**
   Black or African American (PPR 0.17) vs Asian (PPR 0.41) — a 0.24
   absolute gap. Each group individually passes the gate measured against
   overall PPR, but the inter-group gap warrants monitoring in production.
   Likely reflects underlying income distribution in Virginia 2023 ACS
   data rather than model-introduced bias — requires further investigation
   before deployment in high-stakes contexts.

2. **Sex disparity:**
   Male PPR (0.32) vs Female PPR (0.22) — 10 point gap. Both groups
   show strong and nearly equal AUC (0.9456 / 0.9543), indicating the
   model predicts well for both. Disparity reflects income distribution
   in the underlying data.

3. **American Indian alone — small sample warning:**
   Only 68 records in the Virginia test set. Precision 0.375 with
   recall 0.90 indicates the model over-predicts high income for this
   group. AUC 0.9086 is the lowest across all race groups. Metrics
   unreliable at this sample size — national data pull will produce
   more representative results for this group.

4. **Nativity:**
   Foreign born PPR (0.32) slightly higher than native born (0.26).
   AUC nearly identical (0.9488 / 0.9507) — no performance degradation
   across nativity groups.

**Risk Response (NIST AI RMF MANAGE 2.2):**
Gate passed for Virginia development data. The following monitoring
requirements are established for production deployment:
- Re-run fairness audit after national data pull (STATE_CODE="*")
- Monitor per-group PPR drift monthly via Evidently AI
- Flag any group exceeding ±0.15 PPR delta (tighter than gate threshold)
  for human review before next retraining cycle
- Document American Indian group metrics after national pull

---
