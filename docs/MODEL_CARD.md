# Model Card
## responsible-mlops-risk-engine

**Date:** 2026-03-26
**Model:** XGBoost binary classifier
**Registry:** MLflow — `income-risk-xgboost`, alias `staging`
**Pipeline:** responsible-mlops-risk-engine

---

## Model Details

Binary classification model predicting whether an individual's annual
wage income exceeds $75,000. Trained on 2023 American Community Survey
Public Use Microdata Sample — official U.S. Census Bureau microdata.

| Parameter | Value |
|---|---|
| Algorithm | XGBoost (gradient boosted trees) |
| Tuning | Optuna — 30 trials, 5-fold cross-validation |
| n_estimators | 403 |
| max_depth | 5 |
| learning_rate | 0.043 |
| scale_pos_weight | 2.43 |
| Training records | 57,628 (Virginia, FIPS 51) |
| Test records | 14,407 |
| Target | Binary — wage_income >= $75,000 |

---

## Intended Use

**Primary use case:**
Income-based risk scoring for program eligibility screening, resource
allocation analysis, and policy research on labor market income
distributions.

**Intended users:**
Policy analysts, program administrators, and data science teams in
government and public sector organizations requiring auditable,
fairness-enforced income classification.

**Intended context:**
Decision-support tool — model outputs are one input to a human review
process. Not designed for fully automated high-stakes decisions without
human oversight.

---

## Out of Scope Use

The following uses are explicitly out of scope:

- **Automated adverse action** — model output alone is not sufficient
  basis for denying benefits, services, or opportunities without human
  review
- **Individual credit or lending decisions** — not validated for
  financial services regulatory contexts
- **Real-time high-frequency scoring** — current deployment is
  single-instance; auto-scaling not yet implemented
- **Non-U.S. labor markets** — trained exclusively on U.S. Census
  Bureau data; income thresholds and occupation codes are U.S.-specific
- **Historical income prediction** — trained on 2023 data; predictions
  on populations with significantly different labor market conditions
  are unreliable

---

## Training Data

**Source:** American Community Survey (ACS) Public Use Microdata Sample,
2023 1-year estimates. U.S. Census Bureau.

**Scope:** Virginia (FIPS 51) — 88,928 records, working-age adults
(age >= 18), wage earners with valid income records.

**Features used:**

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Individual age in years |
| education | Categorical | Highest educational attainment (encoded) |
| occupation | Categorical | Occupation category (encoded) |
| hours_per_week | Numeric | Usual hours worked per week |
| class_of_worker | Categorical | Employment class (encoded) |
| marital_status | Categorical | Marital status (encoded) |
| person_weight | — | Census sampling weight — sample_weight only, not a feature |

**Features explicitly excluded from model inputs:**
Race, sex, nativity — physically separated at preprocessing, used
exclusively for post-prediction fairness audit. See DL-004, DL-014.

**Train/test split:** 80/20 stratified, RANDOM_STATE=42

---

## Performance

| Metric | Value |
|---|---|
| AUC-ROC | 0.9506 |
| F1 | 0.7633 |
| Precision | 0.6896 |
| Recall | 0.8546 |
| Best CV AUC (Optuna) | 0.9484 |

**Model progression — complexity earned at each stage:**

| Model | AUC-ROC | F1 | Justification |
|---|---|---|---|
| Logistic Regression | 0.9108 | 0.6508 | Baseline established |
| Ridge (L2) | 0.9108 | 0.6507 | No improvement — linear ceiling confirmed |
| XGBoost + Optuna | 0.9506 | 0.7633 | +4.4% AUC, +17.3% F1 — selected |

Full model selection rationale in `docs/decision_log.md` DL-005 through DL-008.

---

## Fairness Audit

**Gate: PASSED** — all groups within ±0.20 positive prediction rate
threshold. Enforced in CI/CD — pipeline exits with code 1 on failure.

**Overall PPR:** 0.2686

| Group | N | PPR | Delta | AUC |
|---|---|---|---|---|
| White alone | 9,685 | 0.279 | 0.010 | 0.954 |
| Black or African American alone | 2,086 | 0.171 | 0.098 | 0.920 |
| American Indian alone | 68 | 0.353 | 0.084 | 0.909 |
| Asian alone | 1,059 | 0.413 | 0.144 | 0.952 |
| Some other race alone | 448 | 0.167 | 0.101 | 0.941 |
| Two or more races | 1,044 | 0.263 | 0.005 | 0.948 |
| Male | 6,929 | 0.321 | 0.053 | 0.946 |
| Female | 7,478 | 0.220 | 0.049 | 0.954 |
| Native born | 12,445 | 0.260 | 0.009 | 0.951 |
| Foreign born | 1,962 | 0.324 | 0.055 | 0.949 |

**Notable findings:**

- Largest inter-group gap: Black or African American (PPR 0.171) vs
  Asian (PPR 0.413) — 0.24 absolute gap. Both groups pass the gate
  individually. Gap reflects income distribution in the underlying
  Virginia 2023 labor market data.
- AUC is consistent across all groups (0.909–0.954) — prediction
  quality does not degrade for any demographic group.
- American Indian group (N=68) — small sample warning. Metrics
  unreliable at this sample size. National data pull required for
  representative results.

Full findings and risk response commitments: `docs/fairness_report.md`

---

## Limitations

**Geographic scope:** Trained on Virginia data. Predictions on
populations in states with significantly different income distributions,
occupation mixes, or demographic compositions may be unreliable.
National retrain is a documented future enhancement — see DL-012.

**Small demographic groups:** American Indian group (N=68 in test set)
has insufficient sample size for reliable metric estimation. Results
for this group should not be treated as definitive.

**Inference contract:** Preprocessing is applied client-side before
endpoint invocation. Feature order, encoding, and scaling must match
the artifacts produced by `src/data/preprocess.py`. Mismatched
preprocessing produces silent prediction errors. See DL-014.

**Class imbalance:** The dataset reflects real-world income distribution
— approximately 22% of records exceed the $75,000 threshold.
scale_pos_weight=2.43 corrects for this during training. Precision
(0.69) reflects residual imbalance.

---

## Monitoring Plan

- **Drift monitoring:** Evidently AI — daily comparison of production
  inputs against training distribution reference. 9 metrics published
  to CloudWatch namespace `ResponsibleRiskEngine/Drift`.
- **Drift threshold:** drift_share > 0.20 triggers CloudWatch alarm
  and manual retraining review — see DL-015.
- **Fairness monitoring:** Per-group PPR monitored monthly. Any group
  exceeding ±0.15 PPR delta flagged for human review before next
  retraining cycle.
- **Retraining:** Manual — human review required before promoting any
  new model version to production. No automated promotion path. See DL-015.

---

## Governance

- NIST AI RMF 1.0 aligned — full mapping in `docs/nist_alignment.md`
- Decision log: DL-001 through DL-016 — `docs/decision_log.md`
- Fairness report: `docs/fairness_report.md`
- All model versions tracked in MLflow registry
- Production promotion requires explicit human sign-off
