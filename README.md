# responsible-mlops-risk-engine

> Production-grade MLOps pipeline for income-based risk scoring using 2023 ACS Census data.
> NIST AI RMF aligned | Demographic Fairness Audits | AWS SageMaker | Terraform IaC

---

## Overview

This project implements a complete, auditable machine learning pipeline for income-based risk scoring using the **2023 American Community Survey (ACS) Public Use Microdata Sample (PUMS)** — official U.S. Census Bureau data representing approximately 3.5 million individuals.

Built to meet the standards expected in federal and government delivery environments, this system emphasizes **responsible AI practices**, **reproducible infrastructure**, and **auditability at every stage** as first-class design requirements.

---

## Background & Design Philosophy

Federal and government ML systems require more than accurate models. They require auditability, fairness documentation, reproducible infrastructure, and alignment with risk management frameworks. This project is built to those standards from day one.

- Model selection documented with metrics and tradeoffs at each stage — Logistic Regression, Ridge, then XGBoost
- Demographic fairness audits integrated into the training and deployment pipeline as a first-class requirement
- Decision log structured for auditor and inspector general review
- Infrastructure versioned and reproducible via Terraform — no manual console steps
- Alignment with NIST AI Risk Management Framework (AI RMF 1.0) from initial design

---

## NIST AI RMF Alignment

This pipeline maps directly to the four NIST AI RMF core functions:

| NIST Function | Implementation in This Project |
|---|---|
| **GOVERN** | Decision log, model cards, stakeholder impact documentation |
| **MAP** | Risk identification across demographic groups, use case scoping |
| **MEASURE** | SHAP explainability, fairness metrics, drift monitoring via Evidently AI |
| **MANAGE** | Automated retraining triggers, champion-challenger deployment, MLflow model registry |

---

## Model Progression — Justified Complexity

Rather than jumping directly to the most complex model, this project follows a principled progression with each transition documented in the **Decision Log** (`docs/decision_log.md`).

Complexity is earned — each stage is only justified if it demonstrates meaningful improvement over the previous one.

| Stage | Model | AUC-ROC | F1 | Notes |
|---|---|---|---|---|
| **Baseline** | Logistic Regression | 0.9108 | 0.6508 | Fully interpretable coefficients — strong baseline |
| **Regularized** | Ridge (L2) | 0.9108 | 0.6507 | CV selected C=100 — no regularization benefit on this dataset |
| **Production** | XGBoost + Optuna | TBD | TBD | Non-linear relationships in occupation and class_of_worker motivate this stage |

Ridge produced no improvement over the baseline — documented in `docs/decision_log.md` DL-006. Linear model ceiling reached at AUC 0.91. XGBoost is evaluated based on near-zero linear coefficients for occupation and class_of_worker, suggesting non-linear signal those features carry that logistic regression cannot capture.

---

## Demographic Fairness Audits

This pipeline treats fairness as an engineering requirement, not a checkbox.

- **Sensitive features tracked separately** — race, sex, and nativity are never used as model inputs but are preserved for fairness analysis throughout the pipeline
- **Disparate impact analysis** — model performance (precision, recall, AUC) computed and compared across demographic groups after every training run
- **SHAP group analysis** — feature importance compared across demographic segments to identify proxy discrimination
- **Audit report generated automatically** — every training run produces a fairness report logged to MLflow and stored in S3

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                           │
│  ACS PUMS 2023 (Census API) → S3 (raw) → S3 (processed)     │
│  DVC tracks all dataset versions                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      TRAINING LAYER                         │
│  Logistic Regression → Ridge → XGBoost  |  Optuna tuning    │
│  MLflow experiment tracking  |  SHAP explainability         │
│  Fairness audit report generated per run                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                       CI/CD LAYER                           │
│  GitHub Actions → lint → test → validate metrics            │
│  Fairness gate → Manual approval → SageMaker deployment     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      SERVING LAYER                          │
│  SageMaker Real-Time Endpoint  |  API Gateway  |  Lambda    │
│  100% data capture to S3 for auditability                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    MONITORING LAYER                         │
│  Evidently AI drift detection (daily)                       │
│  CloudWatch alarms  |  EventBridge → Lambda retraining      │
│  Delayed label evaluation  |  Champion-challenger A/B       │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technology |
|---|---|
| **Cloud** | AWS (SageMaker, S3, Lambda, EventBridge, ECR, CloudWatch, IAM) |
| **IaC** | Terraform v1.5+ |
| **ML Framework** | XGBoost, Scikit-learn |
| **Experiment Tracking** | MLflow |
| **Drift Monitoring** | Evidently AI |
| **Explainability** | SHAP |
| **Hyperparameter Tuning** | Optuna |
| **CI/CD** | GitHub Actions |
| **Data Versioning** | DVC |
| **Demo UI** | Streamlit |
| **Dataset** | ACS PUMS 2023 (U.S. Census Bureau) |

---

## Project Structure

```
responsible-mlops-risk-engine/
├── infrastructure/          # Terraform — all AWS resources
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── modules/
│       ├── sagemaker/
│       ├── monitoring/
│       └── networking/
├── src/
│   ├── data/
│   │   ├── ingest.py        # ACS Census API download
│   │   ├── preprocess.py    # Feature engineering and train/test split
│   │   └── validate.py      # Great Expectations schema checks
│   ├── training/
│   │   ├── baseline.py      # Logistic Regression baseline
│   │   ├── ridge.py         # Ridge Logistic Regression (L2)
│   │   ├── xgboost.py       # XGBoost + Optuna hyperparameter tuning
│   │   ├── evaluate.py      # Metrics + fairness audit
│   │   └── register.py      # MLflow model registry
│   ├── serving/
│   │   ├── inference.py     # SageMaker inference handler
│   │   └── deploy.py        # Endpoint deployment
│   └── monitoring/
│       ├── drift_monitor.py # Evidently AI drift reports
│       └── retrain_trigger.py
├── docs/
│   ├── decision_log.md      # Every model decision justified
│   ├── architecture.md      # System design document
│   ├── fairness_report.md   # Demographic audit findings
│   └── nist_alignment.md   # NIST AI RMF mapping
├── notebooks/               # Exploratory analysis
├── tests/                   # Unit + integration tests
├── .github/workflows/       # CI/CD GitHub Actions
├── .env.example             # Environment variable template
├── config.py                # Central pipeline configuration
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## CI/CD Pipeline

Every push to `main` triggers:

1. **Lint** — flake8 code quality check
2. **Unit tests** — pytest across all modules
3. **Data validation** — schema and null rate checks
4. **Model training** — full pipeline run
5. **Metrics gate** — AUC threshold must be met
6. **Fairness gate** — disparate impact ratio must be within acceptable bounds
7. **Manual approval** — production deployment requires explicit human sign-off
8. **Deploy** — SageMaker endpoint promotion upon approval

In government delivery environments, production deployments require human sign-off regardless of automated test results. A model that passes the AUC gate but fails the fairness audit does not proceed to approval.

---

## Infrastructure as Code

All AWS resources are provisioned via Terraform — nothing is created manually through the console. This ensures:

- Full reproducibility across dev, staging, and production environments
- Infrastructure changes are code-reviewed like application code
- `terraform plan` output is committed to the repo for every change
- `terraform destroy` tears down everything cleanly with one command

```bash
cd infrastructure
terraform init
terraform plan -out=tfplan
```

---

## Dataset

**American Community Survey (ACS) Public Use Microdata Sample (PUMS) — 2023**
Source: U.S. Census Bureau | https://www.census.gov/programs-surveys/acs/microdata.html

Official government microdata representing approximately 3.5 million individuals across the United States. Released annually and used by federal agencies, research institutions, and policy organizations for demographic and economic analysis.

| Feature | Type | Notes |
|---|---|---|
| AGE | Continuous | Age of individual |
| SCHL | Categorical | Educational attainment |
| OCCP | Categorical | Occupation code |
| WKHP | Continuous | Hours worked per week |
| WAGP | Continuous | Wage and salary income — **TARGET SOURCE** |
| COW | Categorical | Class of worker |
| MAR | Categorical | Marital status |
| PWGTP | Continuous | Census sampling weight — XGBoost sample_weight only, not a model input |
| RAC1P | Categorical | Race — **SENSITIVE: fairness audit only** |
| SEX | Categorical | Sex — **SENSITIVE: fairness audit only** |
| NATIVITY | Categorical | Native or foreign born — **SENSITIVE: fairness audit only** |

Sensitive features are never used as model inputs. They are preserved separately for post-prediction fairness analysis only. `PWGTP` is a Census survey methodology artifact — not a personal characteristic. It is passed as `sample_weight` during XGBoost training to produce population-representative predictions.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| ACS PUMS 2023 | Official U.S. Census Bureau microdata — current, government-sourced, 3.5M records |
| Logistic Regression → Ridge → XGBoost | Justified complexity — each transition documented with metrics in decision log |
| person_weight as sample_weight not feature | Census sampling artifact — passed to XGBoost for population representativeness, not used as a model input |
| Fairness gate in CI/CD | Demographic audit is a deployment requirement, not optional |
| Manual approval for SageMaker deploy | Reflects government delivery standard — production requires human sign-off |
| Terraform over CDK | Cloud-agnostic IaC — same workflow applicable to GCP/Azure |
| 100% data capture on endpoint | Government auditability requirement — full prediction history preserved |
| Sensitive features separated at ingest | Prevents proxy discrimination, enables clean fairness reporting |
| config.py as single source of truth | All pipeline parameters in one place — environment-specific secrets in .env |

---

## Deliverables

- [ ] GitHub repo with full CI/CD pipeline
- [ ] Terraform plan output for all AWS infrastructure
- [ ] MLflow experiment comparison — Logistic Regression, Ridge, XGBoost
- [ ] Fairness audit report across demographic groups
- [ ] NIST AI RMF alignment document
- [ ] Decision log structured for auditor review
- [ ] Live SageMaker endpoint (screenshot + curl demo)
- [ ] Evidently AI drift report
- [ ] Streamlit demo application
- [ ] Architecture document

---

## Local Setup

```bash
# Clone the repo
git clone https://github.com/ai-systems-architect/responsible-mlops-risk-engine.git
cd responsible-mlops-risk-engine

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment template and fill in values
cp .env.example .env

# Configure AWS
aws configure

# Run the pipeline
export PYTHONPATH=.
python3 src/data/ingest.py
python3 src/data/preprocess.py
python3 src/training/baseline.py
python3 src/training/ridge.py
python3 src/training/xgboost.py

# Preview infrastructure (no cost)
cd infrastructure && terraform init && terraform plan
```

---

*Built to federal delivery standards. NIST AI RMF 1.0 aligned. Demographic fairness audits built in from day one.*
