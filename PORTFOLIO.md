# responsible-mlops-risk-engine
## End-to-End MLOps Pipeline — Income Risk Scoring

**Raghunath Devayajanam**
GitHub: https://github.com/ai-systems-architect/responsible-mlops-risk-engine

---

## What This Is

A production-grade MLOps pipeline for income-based risk scoring built on
2023 U.S. Census Bureau data. The pipeline covers the full lifecycle —
data ingestion through model deployment, with demographic fairness audits
and drift monitoring built in as first-class requirements.

Built to the standards I apply in federal and government delivery
environments: auditability at every stage, reproducible infrastructure,
and alignment with NIST AI RMF 1.0 from day one.

---

## Why This Dataset

Most income classification projects use the 1994 UCI Adult Census dataset.
That data is 30 years old — income thresholds, occupation distributions,
and demographic compositions are no longer representative of the current
labor market.

The 2023 American Community Survey Public Use Microdata Sample is official
U.S. Census Bureau microdata representing approximately 3.5 million
individuals. It is released annually, used by federal agencies and policy
organizations, and reflects current labor market conditions. The income
threshold in this project — $75,000 — approximates the 2023 U.S. median
household income rather than the 1994 $50K figure still used in most
portfolio reproductions.

---

## Pipeline Architecture

```
Census API → data/raw/ → data/processed/ → model training → evaluation → registry → SageMaker
```

Each stage is a standalone script with a single responsibility.
Parameters live in `config.py` — no hardcoded values in pipeline code.
Sensitive credentials stay in `.env` — never in source code.

**Data layer**
- `src/data/ingest.py` — pulls ACS PUMS 2023 from Census API, saves raw parquet
- `src/data/preprocess.py` — cleans, encodes, scales, splits, separates sensitive features

**Training layer**
- `src/training/baseline.py` — Logistic Regression baseline
- `src/training/ridge.py` — Ridge with cross-validated L2 regularization
- `src/training/train_xgboost.py` — XGBoost with Optuna hyperparameter tuning
- `src/training/evaluate.py` — metrics and fairness audit across demographic groups
- `src/training/register.py` — MLflow model registry

**Serving layer**
- `src/serving/deploy.py` — SageMaker real-time endpoint

**Monitoring layer**
- `src/monitoring/drift_monitor.py` — Evidently AI daily drift detection
- `src/monitoring/retrain_trigger.py` — EventBridge → Lambda retraining

---

## Model Progression

Complexity is earned at each stage — each transition is documented in
`docs/decision_log.md` with the specific metric delta and business rationale.

| Model | AUC-ROC | F1 | Notes |
|---|---|---|---|
| Logistic Regression | 0.9108 | 0.6508 | Strong interpretable baseline |
| Ridge (L2) | 0.9108 | 0.6507 | CV selected C=100 — no regularization benefit |
| XGBoost + Optuna | 0.9506 | 0.7633 | +4.4% AUC, +17.3% F1 over baseline |

Ridge produced no improvement — cross-validation selected the weakest
available regularization, confirming the features are sufficiently
independent. The linear model ceiling at AUC 0.91 was the basis for
moving to XGBoost.

XGBoost confirmed the hypothesis from the baseline coefficients: occupation
and class_of_worker had near-zero linear signal (-0.005 and -0.051) but
ranked 2nd and 4th in XGBoost feature importance. These features carry
non-linear signal — interactions with education level and hours worked —
that logistic regression cannot capture.

**XGBoost best parameters (Optuna, 30 trials):**
n_estimators 403 | max_depth 5 | learning_rate 0.043 | scale_pos_weight 2.43

---

## Fairness Audit

Sensitive features — race, sex, nativity — are physically separated from
model inputs at preprocessing and used exclusively for post-prediction
fairness analysis. The model has no access to protected attributes during
training.

**Fairness gate: PASSED** — all groups within ±0.20 positive prediction
rate threshold.

| Group | PPR | AUC |
|---|---|---|
| White alone | 0.279 | 0.954 |
| Black or African American alone | 0.171 | 0.920 |
| Asian alone | 0.413 | 0.952 |
| Male | 0.321 | 0.946 |
| Female | 0.220 | 0.954 |
| Native born | 0.260 | 0.951 |
| Foreign born | 0.324 | 0.949 |

The largest inter-group gap is between Black or African American (PPR 0.171)
and Asian (PPR 0.413). Both groups pass the gate individually. AUC is
consistent across all groups (0.909–0.954), indicating prediction quality
does not degrade for any demographic. Observed PPR differences reflect
income distribution patterns in the underlying Virginia labor market data.

Full findings and risk response commitments are documented in
`docs/fairness_report.md`.

---

## Responsible AI Design Decisions

**person_weight handling**
The Census Bureau assigns each record a sampling weight representing how
many people that record represents in the full population. Including this
as a model feature would be statistically incorrect — it is a survey
methodology artifact, not a personal characteristic. It is excluded from
model inputs and passed as `sample_weight` to XGBoost, making the model
representative of the full U.S. population rather than just the survey
sample.

**Sensitive feature separation**
Race, sex, and nativity are separated at the preprocessing stage before
any model sees the data. They never pass through encoding or scaling and
are stored in a separate parquet file. This provides a clear audit trail
demonstrating the model never had access to protected attributes.

**Justified complexity**
Each model stage is only justified by demonstrated metric improvement over
the previous one. If XGBoost had not improved meaningfully over Logistic
Regression, the simpler model would have been selected. The decision log
documents this reasoning explicitly — not assumed outcome.

**Fairness gate in CI/CD**
`evaluate.py` exits with code 1 if any demographic group exceeds the PPR
threshold. GitHub Actions fails the pipeline and blocks deployment. A model
that passes the AUC gate but fails the fairness audit does not proceed to
approval.

**Manual deployment approval**
Production deployment requires explicit human sign-off regardless of
automated test results. This reflects the standard in government delivery
environments where a model passing automated gates is necessary but not
sufficient for deployment.

---

## NIST AI RMF Alignment

| Function | Implementation |
|---|---|
| GOVERN | Decision log, fairness report, stakeholder impact documentation |
| MAP | Risk identification across demographic groups, use case scoping |
| MEASURE | SHAP explainability, per-group fairness metrics, Evidently AI drift |
| MANAGE | Automated retraining triggers, MLflow model registry, deployment approval gate |

Full alignment document: `docs/nist_alignment.md`

---

## Infrastructure

All AWS resources provisioned via Terraform — nothing created manually
through the console. Infrastructure changes go through the same code
review process as application code.

Resources provisioned:
- S3 buckets — raw data, processed data, model artifacts
- SageMaker real-time endpoint — ml.m5.xlarge
- EventBridge + Lambda — drift-triggered retraining
- CloudWatch — model performance and fairness drift alarms
- IAM roles — least privilege, no wildcard permissions

---

## Tech Stack

Python 3.9 | XGBoost | Scikit-learn | Optuna | MLflow | Evidently AI |
SHAP | AWS SageMaker | S3 | Lambda | EventBridge | Terraform | GitHub Actions |
Streamlit | pandas | joblib

---

## Dataset

American Community Survey (ACS) Public Use Microdata Sample — 2023
U.S. Census Bureau | https://www.census.gov/programs-surveys/acs/microdata.html

88,928 Virginia records used for development.
National expansion requires one config change: STATE_CODE="*" in config.py pulls ~1.5M records across all 50 states. Virginia is the development baseline; national retrain is the documented production path.

---

*NIST AI RMF 1.0 aligned. Demographic fairness audits built in from day one.*
