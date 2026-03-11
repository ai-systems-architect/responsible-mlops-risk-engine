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

- Model selection documented with metrics and tradeoffs at each stage — OLS, Ridge, then XGBoost
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

Rather than jumping directly to the most complex model, this project follows a principled progression with each transition documented in the **Decision Log** (`docs/decision_log.md`):

| Stage | Model | Why |
|---|---|---|
| **Baseline** | OLS Logistic Regression | Establishes interpretable baseline, fully auditable coefficients |
| **Regularized** | Ridge Regression | Handles multicollinearity in demographic features, reduces overfitting |
| **Production** | XGBoost | Selected based on cross-validated AUC improvement over Ridge — see `docs/decision_log.md` |

Each transition is justified with metrics, tradeoffs, and business rationale — the format used for government client deliverables.

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
│  OLS → Ridge → XGBoost  |  Optuna hyperparameter tuning     │
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
│   │   ├── preprocess.py    # Feature engineering
│   │   └── validate.py      # Great Expectations checks
│   ├── training/
│   │   ├── baseline.py      # OLS logistic regression
│   │   ├── ridge.py         # Ridge regression
│   │   ├── train.py         # XGBoost + Optuna
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

> **Note on deployment:** The final SageMaker deployment step requires manual approval. In government delivery environments, production deployments require human sign-off regardless of automated test results. A model that passes the AUC gate but fails the fairness audit does not proceed to approval.

---

## Infrastructure as Code

All AWS resources are provisioned via Terraform — nothing is created manually through the console. This ensures:

- Full reproducibility across dev, staging, and production environments
- Infrastructure changes are code-reviewed like application code
- `terraform plan` output is committed to the repo for every change
- `terraform destroy` tears down everything cleanly with one command

To preview all infrastructure without deploying:
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
| WAGP | Continuous | Wage and salary income — **TARGET** |
| COW | Categorical | Class of worker |
| MAR | Categorical | Marital status |
| RAC1P | Categorical | Race — **SENSITIVE: fairness audit only** |
| SEX | Categorical | Sex — **SENSITIVE: fairness audit only** |
| NATIVITY | Categorical | Native or foreign born — **SENSITIVE: fairness audit only** |

Sensitive features are never used as model inputs. They are preserved separately for post-prediction fairness analysis only.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| ACS PUMS 2023 | Official U.S. Census Bureau microdata — current, government-sourced, 3.5M records |
| OLS → Ridge → XGBoost progression | Justified complexity — each step documented with metrics in decision log |
| Fairness gate in CI/CD | Demographic audit is a deployment requirement, not optional |
| Manual approval for SageMaker deploy | Reflects government delivery standard — production requires human sign-off |
| Terraform over CDK | Cloud-agnostic IaC — same workflow applicable to GCP/Azure |
| 100% data capture on endpoint | Government auditability requirement — full prediction history preserved |
| Sensitive features separated at ingest | Prevents proxy discrimination, enables clean fairness reporting |

---

## Deliverables

- [ ] GitHub repo with full CI/CD pipeline
- [ ] Terraform plan output for all AWS infrastructure
- [ ] MLflow experiment comparison across OLS, Ridge, XGBoost
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

# Configure AWS
aws configure

# Preview infrastructure (no cost)
cd infrastructure && terraform init && terraform plan
```

---

*Built to federal delivery standards. NIST AI RMF 1.0 aligned. Demographic fairness audits built in from day one.*
