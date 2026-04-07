# responsible-mlops-risk-engine

> Production-grade MLOps pipeline for income-based risk scoring on 2023 U.S. Census data.
> NIST AI RMF 1.0 aligned | Demographic fairness audits | AWS SageMaker | Terraform IaC

![Visitors](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fai-systems-architect%2Fresponsible-mlops-risk-engine&count_bg=%230D1117&title_bg=%23555555&icon=github.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=false)

---

## What This Is

A complete, auditable MLOps pipeline for income-based risk scoring built on
the 2023 American Community Survey Public Use Microdata Sample — official
U.S. Census Bureau data. The pipeline covers the full lifecycle: data
ingestion through model deployment, with demographic fairness enforcement
and drift monitoring built in as first-class requirements.

Built to the standards applied in federal and government delivery
environments: auditability at every stage, reproducible infrastructure,
and alignment with NIST AI RMF 1.0 from day one.

---

## Why This Dataset

The 2023 ACS PUMS was a deliberate choice. It is official Census Bureau
microdata released annually — current, government-sourced, and representative
of today's labor market. The income threshold is set at $75,000, approximating
the 2023 U.S. median household income, grounding the classification task in
present economic reality.

---

## Model Progression

Complexity is earned at each stage. Every transition is documented in
`docs/decision_log.md` with the specific metric delta that justified it.

| Stage | Model | AUC-ROC | F1 | Notes |
|---|---|---|---|---|
| Baseline | Logistic Regression | 0.9108 | 0.6508 | Strong interpretable baseline |
| Regularized | Ridge (L2) | 0.9108 | 0.6507 | CV selected C=100 — no regularization benefit |
| Production | XGBoost + Optuna | 0.9506 | 0.7633 | +4.4% AUC, +17.3% F1 over baseline |

Ridge produced no improvement — cross-validation selected the weakest
available regularization, confirming the features are sufficiently
independent. The linear model ceiling at AUC 0.91 justified moving to
XGBoost.

XGBoost confirmed the non-linear signal hypothesis from the baseline:
`occupation` had near-zero linear signal (-0.005) but ranked 2nd in
XGBoost feature importance (0.17). `class_of_worker` showed the same
pattern. These features carry interaction effects that logistic regression
cannot capture.

XGBoost best parameters (Optuna, 30 trials): n_estimators 403, max_depth 5,
learning_rate 0.043, scale_pos_weight 2.43.

---

## Fairness Audit

Sensitive features — race, sex, nativity — are physically separated from
model inputs at preprocessing and used exclusively for post-prediction
fairness analysis. The model has no access to protected attributes.

**Gate: PASSED** — all groups within ±0.20 positive prediction rate threshold.
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

The fairness gate is enforced in CI/CD — `evaluate.py` exits with code 1
on failure, blocking all downstream deployment steps. Full findings and
risk response commitments in `docs/fairness_report.md`.

---

## NIST AI RMF 1.0 Alignment

| Function | Implementation |
|---|---|
| GOVERN | Decision log DL-001 to DL-016, sensitive feature separation, MLflow lineage |
| MAP | Risk identification table, use case scoping, impact assessment per demographic group |
| MEASURE | Per-group PPR and AUC, fairness gate in CI/CD, Evidently AI drift monitoring |
| MANAGE | Multi-stage deployment approval, retraining triggers, CloudWatch alarms |

Full control mapping in `docs/nist_alignment.md`.

---

## Architecture

```
Census API → ingest.py → data/raw/
                │
          preprocess.py → data/processed/
                │
    ┌───────────┴───────────┐
    │                       │
 X_train/y_train      X_test/y_test + sensitive_*.parquet
    │
baseline.py → ridge.py → train_xgboost.py (Optuna)
                                │
                          models/*.joblib
                          models/*.json (native — SageMaker)
                                │
                          evaluate.py (metrics + fairness gate)
                                │
                          register.py (MLflow registry)
                                │
                          deploy.py → S3 → SageMaker endpoint
                                │
                    drift_monitor.py → Evidently AI → CloudWatch
```

---

## Infrastructure

All AWS resources provisioned via Terraform — nothing created manually
through the console. `infrastructure/main.tf` provisions S3 buckets,
SageMaker IAM role with least-privilege policy, and CloudWatch alarms
for endpoint availability, error rate, and p99 latency.

```bash
cd infrastructure
terraform init
terraform plan -var="aws_account_id=YOUR_ACCOUNT_ID"
terraform apply -var="aws_account_id=YOUR_ACCOUNT_ID"
```

Standing cost: ~$0.30/month (CloudWatch alarms only). SageMaker endpoint
costs ~$5/day — deployed on demand and destroyed immediately after use.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| ACS PUMS 2023 | Official Census Bureau microdata — current, annually updated, income threshold and demographics grounded in 2023 |
| person_weight as sample_weight | Census sampling artifact — passed to XGBoost for population representativeness, not a model input |
| Fairness gate in CI/CD | Demographic audit is a deployment requirement, not a report |
| Native XGBoost JSON format for SageMaker | Eliminates container script-loading issues — booster loaded directly, no custom inference script needed |
| SageMaker SDK pinned to 2.x | SDK 3.x is a complete API rewrite released early 2026 — too new for production use |
| Virginia for development | 88,928 records — fast iteration. National retrain is one config change: STATE_CODE="*" |
| Manual deployment approval | Reflects government delivery standard — CI/CD gates are necessary but not sufficient |
| config.py as single source of truth | All parameters in one place — no hardcoded values in pipeline code |

---

## Project Structure

```
responsible-mlops-risk-engine/
├── config.py                     # All pipeline parameters
├── .env.example                  # Credential template
├── requirements.txt              # Dependencies
├── PORTFOLIO.md                  # Portfolio overview
├── src/
│   ├── data/
│   │   ├── ingest.py             # ACS Census API pull
│   │   └── preprocess.py        # Feature engineering, split
│   ├── training/
│   │   ├── baseline.py          # Logistic Regression
│   │   ├── ridge.py             # Ridge with L2
│   │   ├── train_xgboost.py     # XGBoost + Optuna
│   │   ├── evaluate.py          # Metrics + fairness gate
│   │   └── register.py          # MLflow registry
│   ├── serving/
│   │   └── deploy.py            # SageMaker deployment
│   └── monitoring/
│       └── drift_monitor.py     # Evidently AI + CloudWatch
├── infrastructure/
│   ├── main.tf                  # S3, IAM, CloudWatch
│   ├── variables.tf
│   └── outputs.tf
└── docs/
    ├── decision_log.md          # DL-001 through DL-016
    ├── fairness_report.md       # Stakeholder fairness audit
    ├── nist_alignment.md        # NIST AI RMF 1.0 mapping
    ├── architecture.md          # System design
    ├── model_card.md            # Model details, intended use, limitations
    └── runbook.md               # Operational procedures
```

---

## Deliverables

- ✅ End-to-end MLOps pipeline — ingest through deployment
- ✅ Three-model progression with documented justification
- ✅ Fairness audit — 10 demographic groups, CI/CD gate
- ✅ MLflow experiment tracking — full artifact bundle
- ✅ Terraform IaC — S3, IAM, CloudWatch
- ✅ SageMaker real-time endpoint — deployed and verified
- ✅ Evidently AI drift monitoring — CloudWatch metrics
- ✅ NIST AI RMF 1.0 alignment document
- ✅ Decision log — DL-001 through DL-016
- ✅ Model card — intended use, limitations, fairness summary
- ✅ Runbook — deployment, rollback, drift response, retraining
- ✅ Streamlit demo

---

## Local Setup

```bash
git clone https://github.com/ai-systems-architect/responsible-mlops-risk-engine.git
cd responsible-mlops-risk-engine

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in CENSUS_API_KEY, AWS_ACCOUNT_ID, S3_BUCKET, SAGEMAKER_ROLE_ARN

aws configure
export PYTHONPATH=.

# Run the pipeline
python3 src/data/ingest.py
python3 src/data/preprocess.py
python3 src/training/baseline.py
python3 src/training/ridge.py
python3 src/training/train_xgboost.py
python3 src/training/evaluate.py
python3 src/training/register.py  # requires: mlflow ui running

# Infrastructure (no cost until apply)
cd infrastructure
terraform init
terraform plan -var="aws_account_id=YOUR_ACCOUNT_ID"
```

---

## Dataset

American Community Survey (ACS) Public Use Microdata Sample — 2023
U.S. Census Bureau | https://www.census.gov/programs-surveys/acs/microdata.html

88,928 Virginia records used for development. National expansion requires
one config change: `STATE_CODE = "*"` in `config.py`.

---

## Tech Stack

Python 3.9 | XGBoost | Scikit-learn | Optuna | MLflow | Evidently AI |
SHAP (planned) | AWS SageMaker | S3 | CloudWatch | IAM | Terraform | GitHub Actions | Streamlit

---

*NIST AI RMF 1.0 aligned. Demographic fairness audits built in from day one.*
