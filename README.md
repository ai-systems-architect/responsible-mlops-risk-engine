# responsible-mlops-risk-engine

> Production-grade MLOps pipeline for income-based risk scoring using 2023 ACS Census data.
> NIST AI RMF aligned | Demographic Fairness Audits | AWS SageMaker | Terraform IaC

---

## Overview

This project implements a complete, auditable machine learning pipeline for income-based risk scoring using the **2023 American Community Survey (ACS) Public Use Microdata Sample (PUMS)** — official U.S. Census Bureau data representing approximately 3.5 million individuals.

Built to meet the standards expected in federal and government delivery environments, this system emphasizes **responsible AI practices**, **reproducible infrastructure**, and **auditability at every stage** — not as afterthoughts, but as first-class design requirements.

---

## Why This Project Exists

Most ML portfolio projects train a model and stop. This project simulates what a government delivery team actually ships:

- A **justified model selection process** — not just "I picked XGBoost"
- A **fairness audit** built into the pipeline — not retrofitted after the fact
- Infrastructure that can be **reviewed, versioned, and reproduced** by any team member
- A **decision log** structured for auditors, inspectors general, and program managers
- Alignment with **NIST AI Risk Management Framework (AI RMF 1.0)**

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

Rather than jumping directly to the most complex model, this project follows a principled progression:

| Stage | Model | Why |
|---|---|---|
| **Baseline** | OLS Logistic Regression | Establishes interpretable baseline, fully auditable coefficients |
| **Regularized** | Ridge Regression | Handles multicollinearity in demographic features, reduces overfitting |
| **Production** | XGBoost | Justified by AUC improvement >8% over Ridge, with SHAP to maintain explainability |

Each transition is documented in the **Decision Log** (`docs/decision_log.md`) with metrics, tradeoffs, and justification — the format used for government client deliverables.

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
│  ACS PUMS 2023 (Census API) → S3 (raw) → S3 (processed)   │
│  DVC tracks all dataset versions                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      TRAINING LAYER                         │
│  OLS → Ridge → XGBoost  |  Optuna hyperparameter tuning    │
│  MLflow experiment tracking  |  SHAP explainability        │
│  Fairness audit report generated per run                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                       CI/CD LAYER                           │
│  GitHub Actions → lint → test → validate metrics           │
│  Auto-promote to SageMaker if AUC > 0.82 + fairness pass   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      SERVING LAYER                          │
│  SageMaker Real-Time Endpoint  |  API Gateway  |  Lambda   │
│  100% data capture to S3 for auditability                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    MONITORING LAYER                         │
│  Evidently AI drift detection (daily)                       │
│  CloudWatch alarms  |  EventBridge → Lambda retraining     │
│  Delayed label evaluation  |  Champion-challenger A/B      │
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
│   └── nist_alignment.md    # NIST AI RMF mapping
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
5. **Metrics gate** — AUC > 0.82 AND fairness audit pass required
6. **Fairness gate** — disparate impact ratio must be within acceptable bounds
7. **Deploy** — automatic promotion to SageMaker if all gates pass

The fairness gate is intentional — a model that passes AUC but fails the demographic audit does **not** get deployed. This is the government delivery standard.

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
| ACS PUMS 2023 over UCI 1994 Census | Current government data — 30 years more recent, 70x more records |
| OLS → Ridge → XGBoost progression | Justified complexity — each step documented with metrics |
| Fairness gate in CI/CD | Demographic audit is a deployment requirement, not optional |
| Terraform over CDK | Cloud-agnostic IaC — same workflow applicable to GCP/Azure |
| 100% data capture on endpoint | Government auditability requirement — full prediction history |
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
