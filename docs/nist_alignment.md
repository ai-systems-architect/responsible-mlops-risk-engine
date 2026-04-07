# NIST AI RMF 1.0 Alignment
## responsible-mlops-risk-engine

**Date:** 2026-03-19
**Framework:** NIST AI Risk Management Framework 1.0
**Reference:** [NIST AI RMF](https://airc.nist.gov/RMF)

This document maps each component of the pipeline to specific NIST AI RMF
controls.

---

## Framework Overview

The NIST AI RMF organizes AI risk management into four functions:

| Function | Purpose |
|---|---|
| **GOVERN** | Establish policies, accountability, and culture for AI risk management |
| **MAP** | Identify and classify AI risks in context |
| **MEASURE** | Analyze and assess AI risks with quantitative and qualitative methods |
| **MANAGE** | Prioritize and address AI risks with documented responses |

---

## GOVERN

### GOVERN 1.1 — Policies and Procedures
**Control:** Organizational policies establish expectations for responsible AI.

**Implementation:**
The pipeline enforces responsible AI requirements through code rather than
policy documents alone. The fairness gate in `evaluate.py` blocks deployment
if any demographic group exceeds the ±0.20 PPR threshold. This converts a
policy commitment into an automated enforcement mechanism that cannot be
bypassed without explicit code changes.

The decision log (`docs/decision_log.md`) records every design choice with
rationale — DL-001 through DL-012 — providing an auditable trail of
deliberate decisions rather than undocumented assumptions.

---

### GOVERN 1.4 — Model Versioning and Lineage
**Control:** AI system versions and provenance are documented and traceable.

**Implementation:**
MLflow experiment tracking (`src/training/register.py`) captures the complete
lineage of every model version:
- All hyperparameters — n_estimators, max_depth, learning_rate, scale_pos_weight
- Training dataset — ACS PUMS 2023, Virginia (FIPS 51), 57,628 records
- Performance metrics — AUC-ROC, F1, precision, recall
- Fairness metrics — per-group PPR and AUC across race, sex, nativity
- Preprocessing artifacts — encoders and scaler attached to the run

Model `income-risk-xgboost v2` is registered in the MLflow Model Registry
with alias `staging`. Promotion to production requires explicit human approval —
no automated promotion path exists.

---

### GOVERN 1.7 — Sensitive Attribute Handling
**Control:** Protected attributes are handled with documented safeguards.

**Implementation:**
Race, sex, and nativity are physically separated from model inputs at the
preprocessing stage in `src/data/preprocess.py`. They are stored in a
separate parquet file (`data/processed/sensitive_*.parquet`) and never pass
through encoding, scaling, or model training. The model has no direct access
to protected attributes at any stage.

Sensitive features are rejoined exclusively for post-prediction fairness
auditing in `src/training/evaluate.py`. This provides a clear audit trail
demonstrating the separation was maintained throughout the pipeline.

---

## MAP

### MAP 1.1 — Use Case Scoping
**Control:** The AI system's intended use, affected populations, and deployment
context are clearly defined.

**Implementation:**
This pipeline is scoped to income-based risk scoring using 2023 U.S. Census
Bureau American Community Survey data. The income threshold of $75,000
approximates the 2023 U.S. median household income and is documented in
`config.py` and `docs/decision_log.md` DL-002.

Affected populations: U.S. working-age adults (18+) across all demographic
groups represented in ACS PUMS data. Virginia (FIPS 51) used for development
and production. National expansion (`STATE_CODE="*"`) is a documented future
enhancement.

Deployment context: The model produces probability scores — not binary
decisions. Human review is required before any consequential action is taken
on model output.

---

### MAP 2.1 — Risk Identification
**Control:** AI risks are identified across the full lifecycle.

**Implementation:**
The following risks were identified and addressed during development:

| Risk | Mitigation |
|---|---|
| Demographic bias in predictions | Fairness audit across race, sex, nativity — gate blocks deployment on failure |
| Sampling artifact as model feature | person_weight excluded from model inputs — used as sample_weight only |
| Training data leakage into fairness audit | Sensitive features separated before train/test split |
| Model degradation over time | Evidently AI drift monitoring — daily comparison against training distribution |
| Irreproducible results | RANDOM_STATE=42 fixed across all splits, CV, and model initialization |
| Stale model in production | MLflow Model Registry with explicit staging and production promotion gates |

---

### MAP 5.1 — Impact Assessment
**Control:** Potential impacts on individuals and communities are assessed.

**Implementation:**
The fairness audit (`src/training/evaluate.py`) quantifies differential impact
across demographic groups. Key findings from the Virginia development audit:

- Black or African American PPR (0.171) vs Asian PPR (0.413) — largest
  inter-group gap at 0.242. Both pass the gate measured against overall PPR.
  Reflects income distribution in underlying Virginia labor market data.
- American Indian group (n=68) — flagged for small sample size. Metrics
  unreliable at this scale. Priority review required after national data pull.
- Male PPR (0.321) vs Female PPR (0.220) — reflects wage gap in underlying
  data. Model AUC equivalent across sexes (0.946 / 0.954).

Full findings documented in `docs/fairness_report.md`.

---

## MEASURE

### MEASURE 2.5 — Bias and Fairness Evaluation
**Control:** Bias is evaluated across demographic groups using quantitative methods.

**Implementation:**
`src/training/evaluate.py` computes the following per demographic group:
- Positive Prediction Rate (PPR) — fraction predicted high income
- PPR delta — deviation from overall PPR (0.2686)
- AUC-ROC — model discrimination ability within each group
- Precision, recall, F1 — classification quality per group

Groups evaluated: race (6 groups), sex (2 groups), nativity (2 groups).
Groups with fewer than 30 records excluded — metrics unreliable at small
sample sizes.

**Fairness gate:** Models where any group's PPR deviates from overall PPR
by more than ±0.20 do not proceed to deployment approval. The gate is
enforced in CI/CD — `evaluate.py` exits with code 1 on failure, blocking
the pipeline.

**Virginia audit result: PASSED** — 0 / 10 groups exceeded threshold.

---

### MEASURE 2.6 — Model Performance Monitoring
**Control:** Model performance is monitored post-deployment with defined thresholds.

**Implementation:**
`src/monitoring/drift_monitor.py` runs Evidently AI drift analysis comparing:
- Reference: X_train (training distribution)
- Current: incoming production inference data (simulated by X_test)

Statistical tests applied:
- Wasserstein distance (normed) — continuous features: age, hours_per_week
- Jensen-Shannon distance — categorical features: marital_status
- Chi-squared — other categorical features

Metrics published to CloudWatch namespace `ResponsibleRiskEngine/Drift`:
- `DriftShare` — fraction of features drifted
- `DriftedFeatures` — count
- `DatasetDrift` — binary flag
- Per-feature drift flags for all 6 model features

CloudWatch alarms defined in `infrastructure/main.tf` trigger alerts on
endpoint availability, invocation errors, and latency degradation.

**Virginia baseline result:** 0 / 6 features drifted. Drift share: 0.0.

---

### MEASURE 2.7 — Fairness Gate Enforcement
**Control:** Fairness thresholds are enforced programmatically, not just reported.

**Implementation:**
The fairness gate is a hard stop in the deployment pipeline. `evaluate.py`
returns exit code 1 if any group exceeds the PPR threshold. GitHub Actions
treats this as a pipeline failure and blocks all subsequent deployment steps.

The fairness threshold (±0.20 PPR) is defined in `config.py` as
`FAIRNESS_THRESHOLD` — a single value that controls both the audit logic
and the gate enforcement. Changing the threshold requires a code change,
pull request, and CI/CD rerun.

---

### MEASURE 4.1 — Explainability
**Control:** Model predictions can be explained to affected individuals and
oversight bodies.

**Implementation:**
Feature importance from XGBoost training:
- `hours_per_week` — 0.4397 (dominant predictor)
- `occupation` — 0.1719
- `education` — 0.1403
- `class_of_worker` — 0.1132
- `age` — 0.0713
- `marital_status` — 0.0636

The non-linear signal in occupation and class_of_worker — near-zero in
logistic regression coefficients but ranked 2nd and 4th in XGBoost feature
importance — is documented in `docs/decision_log.md` DL-008 and justified
the move from the interpretable baseline to XGBoost.

SHAP (SHapley Additive exPlanations) is implemented at two levels:

- **Global explainability** — `src/training/evaluate.py` generates a beeswarm
  plot (`docs/shap_summary.png`) across all test records after the fairness
  gate passes. Each dot is one record; color encodes feature value magnitude;
  x-axis shows SHAP impact on prediction. Runs as part of the standard
  evaluation pipeline.

- **Per-record explainability** — the Streamlit demo renders a SHAP waterfall
  plot for every prediction. The waterfall shows which features pushed the
  specific prediction above or below the model baseline, answering the
  question a reviewer or auditor would ask: why did the model score this
  individual this way?

Both use `shap.TreeExplainer` — the exact computation method for gradient
boosted trees, not an approximation.

---

## MANAGE

### MANAGE 1.3 — Deployment Approval Gate
**Control:** AI systems require explicit approval before production deployment.

**Implementation:**
The deployment pipeline enforces a multi-stage approval process:

1. `evaluate.py` — AUC must exceed 0.82 (`MIN_AUC_THRESHOLD`) and fairness
   gate must pass. Both are automated.
2. MLflow Model Registry — model must be registered and in `staging` alias
   before deployment. Registry update is manual.
3. `deploy.py` — deployment requires explicit execution. No automated
   promotion from staging to production exists. A human must run the script.
4. SageMaker endpoint — `deploy.py` pauses before destroying the endpoint
   and requires explicit confirmation, providing a final review window.

---

### MANAGE 2.2 — Risk Response Documentation
**Control:** Risk findings are documented with response commitments.

**Implementation:**
`docs/fairness_report.md` documents all fairness audit findings with explicit
risk response commitments:

- Inter-group race gap (Black 0.171 vs Asian 0.413) — documented, monitored
- American Indian small sample (n=68) — flagged for priority review after
  national data pull
- Sex disparity (Male 0.321 vs Female 0.220) — reflects underlying data,
  model AUC equivalent across sexes
- Monitoring threshold set at ±0.15 PPR delta — tighter than gate, for
  early warning before retraining is required

---

### MANAGE 2.4 — Ongoing Monitoring
**Control:** AI systems are monitored post-deployment with defined response plans.

**Implementation:**
`src/monitoring/drift_monitor.py` provides ongoing monitoring:
- Daily drift detection against training distribution
- 9 CloudWatch metrics published per run
- Alert threshold: drift_share > 0.20 triggers retraining recommendation
- Per-feature drift flags enable targeted investigation

CloudWatch alarms (`infrastructure/main.tf`):
- `endpoint-availability` — fires if invocations drop to zero
- `invocation-errors` — fires if >5% of requests return errors
- `model-latency` — fires if p99 latency exceeds 2000ms

Fairness drift monitored separately — per-group PPR tracked against
baseline values from `docs/fairness_report.md`.

---

### MANAGE 4.1 — Retraining Triggers
**Control:** Conditions that trigger model retraining are defined and documented.

**Implementation:**
The following conditions trigger retraining:

| Trigger | Threshold | Detection Method |
|---|---|---|
| Feature drift | drift_share > 0.20 | drift_monitor.py → CloudWatch |
| AUC degradation | AUC drops below 0.82 | evaluate.py gate |
| Fairness drift | Any group PPR delta > 0.15 | drift_monitor.py |
| Data staleness | Annual ACS data release | Scheduled — config.py ACS_YEAR |

Retraining requires running the full pipeline from ingestion:
```bash
python3 src/data/ingest.py
python3 src/data/preprocess.py
python3 src/training/train_xgboost.py
python3 src/training/evaluate.py
python3 src/training/register.py
python3 src/serving/deploy.py
```

National data pull activated by setting `STATE_CODE = "*"` in `config.py`.

---

## Summary Table

| NIST Control | Pipeline Component |
|---|---|
| GOVERN 1.1 | Fairness gate in evaluate.py, decision log |
| GOVERN 1.4 | MLflow model registry, run lineage |
| GOVERN 1.7 | Sensitive feature separation in preprocess.py |
| MAP 1.1 | Use case scoped, income threshold documented |
| MAP 2.1 | Risk identification table, mitigations implemented |
| MAP 5.1 | Per-group impact assessment in fairness_report.md |
| MEASURE 2.5 | evaluate.py — PPR, AUC, F1 per demographic group |
| MEASURE 2.6 | drift_monitor.py — Evidently AI + CloudWatch |
| MEASURE 2.7 | Fairness gate enforced in CI/CD pipeline |
| MEASURE 4.1 | XGBoost feature importance, SHAP global beeswarm + per-record waterfall |
| MANAGE 1.3 | Multi-stage deployment approval gate |
| MANAGE 2.2 | fairness_report.md — findings + response commitments |
| MANAGE 2.4 | drift_monitor.py + CloudWatch alarms in Terraform |
| MANAGE 4.1 | Retraining triggers defined with thresholds |

---

*This document reflects the Virginia (FIPS 51) development and production
pipeline. National expansion to STATE_CODE="*" is a documented future
enhancement. A full NIST alignment review is recommended before national
deployment.*
