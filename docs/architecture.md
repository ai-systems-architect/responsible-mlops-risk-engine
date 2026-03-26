# System Architecture
## responsible-mlops-risk-engine

**Date:** 2026-03-19
**Author:** Raghunath Devayajanam

---

## Overview

End-to-end MLOps pipeline for income-based risk scoring on U.S. Census
Bureau data. Built to production standards — reproducible infrastructure,
automated fairness enforcement, drift monitoring, and a documented path
from development to national deployment.

The architecture is intentionally layered. Each layer has a single
responsibility and can be replaced independently. Swapping SageMaker for
Kubernetes or adding a new data source doesn't require touching the training
or fairness layers.

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│                                                                     │
│  Census API ──► ingest.py ──► data/raw/         (parquet)          │
│                     │                                               │
│                     ▼                                               │
│             preprocess.py ──► data/processed/   (parquet, joblib)  │
│                     │                                               │
│          ┌──────────┴──────────┐                                   │
│          │                     │                                   │
│    X_train/y_train        X_test/y_test + sensitive_*.parquet      │
└──────────┼─────────────────────┼───────────────────────────────────┘
           │                     │
┌──────────▼─────────────────────▼───────────────────────────────────┐
│                       TRAINING LAYER                                │
│                                                                     │
│  baseline.py  ──►  ridge.py  ──►  train_xgboost.py (Optuna)       │
│                                          │                          │
│                                    models/*.joblib                  │
│                                    models/*.json (native)           │
│                                          │                          │
│                                   evaluate.py                       │
│                                   (metrics + fairness gate)         │
│                                          │                          │
│                                   register.py                       │
│                                   (MLflow registry)                 │
└──────────────────────────────────────────────────────────────────── ┘
           │
┌──────────▼──────────────────────────────────────────────────────────┐
│                      SERVING LAYER                                   │
│                                                                      │
│  deploy.py ──► S3 (model.tar.gz) ──► SageMaker Endpoint            │
│                                      (XGBoost container)            │
│                                             │                        │
│                                      HTTPS invocations              │
│                    https://runtime.sagemaker.{region}.amazonaws.com │
│                    /endpoints/{endpoint-name}/invocations           │
└──────────────────────────────────────────────────────────────────── ┘
           │
┌──────────▼──────────────────────────────────────────────────────────┐
│                     MONITORING LAYER                                 │
│                                                                      │
│  drift_monitor.py ──► Evidently AI ──► CloudWatch metrics          │
│                              │                                       │
│                       drift_report_*.html                           │
│                              │                                       │
│                    CloudWatch Alarms (Terraform)                     │
│                    - endpoint-availability                           │
│                    - invocation-errors                               │
│                    - model-latency                                   │
│                    - DriftShare / DriftedFeatures                    │
└──────────────────────────────────────────────────────────────────── ┘
```

---

## Data Flow

### Ingestion
`src/data/ingest.py` pulls ACS PUMS 2023 microdata from the U.S. Census
Bureau API. The pull is parameterized by `STATE_CODE` in `config.py`:
- `"51"` — Virginia, 88,928 records, used for development
- `"*"` — all 50 states, ~1.5M records, the production path

Raw data is saved as timestamped parquet to `data/raw/`. The Census API
key is loaded from `.env` — never stored in source code.

### Preprocessing
`src/data/preprocess.py` handles all feature engineering in a single pass:

1. Drop invalid rows — negative income, age below 18
2. Create binary target — `wage_income >= 75000` → `high_income`
3. Separate sensitive features — race, sex, nativity physically removed
   from model data and saved separately
4. Encode categoricals — LabelEncoder fitted on training data
5. Scale numerics — StandardScaler fitted on training data
6. Stratified train/test split — 80/20, RANDOM_STATE=42

Encoders and scaler are saved to `data/processed/` alongside the split
data. These artifacts must be loaded together with the model at inference
time — applying different preprocessing to new data produces meaningless
predictions.

**person_weight handling:** The Census Bureau assigns each record a
population weight representing how many people that record represents.
It is excluded from model features and passed as `sample_weight` to
XGBoost, making the model representative of the full U.S. population
rather than the survey sample.

---

## Training

Three models were trained in sequence. Complexity is earned — each stage
required demonstrated metric improvement to justify the added complexity.

### Logistic Regression (baseline)
AUC 0.9108, F1 0.6508. Strong linear baseline. Coefficients reveal
`hours_per_week` as the dominant predictor (1.72), while `occupation`
carries near-zero linear signal (-0.005).

### Ridge Logistic Regression
AUC 0.9108, F1 0.6507. Cross-validation selected C=100 — the weakest
available regularization — confirming features are sufficiently independent
and L2 provides no benefit. Linear ceiling confirmed at AUC 0.91.

### XGBoost + Optuna
AUC 0.9506 (+0.0398), F1 0.7633 (+0.1125). 30 Optuna trials over 5-fold
cross-validation. The non-linear signal hypothesis from the baseline was
confirmed — `occupation` jumped from near-zero linear coefficient to 2nd
most important feature (0.17) in XGBoost. `class_of_worker` showed the
same pattern. These features carry interaction effects with education and
hours worked that logistic regression cannot capture.

Best parameters: n_estimators 403, max_depth 5, learning_rate 0.043,
scale_pos_weight 2.43 (handles class imbalance).

---

## Fairness Enforcement

Sensitive features — race, sex, nativity — are physically separated from
model inputs before training and used exclusively for post-prediction audit.
The model has no access to protected attributes.

`src/training/evaluate.py` computes positive prediction rate (PPR) and
AUC-ROC for each demographic group. Groups deviating more than ±0.20 from
the overall PPR fail the fairness gate, and the pipeline exits with code 1,
blocking deployment.

**Virginia audit result: PASSED** — 0/10 groups exceeded threshold.
Full findings in `docs/fairness_report.md`.

---

## MLflow Experiment Tracking

`src/training/register.py` logs the full artifact bundle to MLflow:
- All hyperparameters
- Performance metrics and per-group fairness metrics
- Model artifact with input/output signature
- Preprocessing artifacts — encoders and scaler
- Fairness report attached to the run

Model registered as `income-risk-xgboost v2` with alias `staging`.
Production promotion requires explicit human approval — no automated
promotion path exists.

MLflow runs locally — `mlruns/` is gitignored. Start the UI with:
```bash
mlflow ui  # http://localhost:5000
```

---

## Infrastructure

All AWS resources are provisioned via Terraform — nothing created manually
through the console. `infrastructure/main.tf` provisions:

| Resource | Purpose |
|---|---|
| S3 raw data | Stores raw ACS PUMS parquet files from Census API pull |
| S3 processed | Stores engineered features, encoders, scaler artifacts |
| S3 models | Stores trained model artifacts for SageMaker deployment |
| IAM role | Least-privilege SageMaker execution role |
| IAM policy | Scoped to project buckets + default SageMaker bucket |
| CloudWatch alarms | Endpoint availability, error rate, p99 latency |

```bash
cd infrastructure
terraform init
terraform apply -var="aws_account_id=YOUR_ACCOUNT_ID"
```

Cost: ~$0.30/month standing (CloudWatch alarms only).

---

## Serving

`src/serving/deploy.py` packages the model in XGBoost native JSON format,
uploads to S3, and deploys to a SageMaker real-time endpoint.

**Why native JSON format:** The SageMaker XGBoost container natively loads
XGBoost's JSON format without requiring a custom inference script. Using a
custom script triggered the container's pip install mechanism, which failed
consistently regardless of script naming. Saving the booster directly with
`model.get_booster().save_model()` bypasses this entirely.

Endpoint URL:
```
https://runtime.sagemaker.{region}.amazonaws.com
/endpoints/{endpoint-name}/invocations
```

The endpoint costs ~$5/day. In this portfolio it is deployed for
verification and destroyed immediately after screenshot. Production
deployment would use auto-scaling to bring instance count to zero during
off-peak hours.

Preprocessing is applied client-side before invocation. See DL-014 for
the full architectural rationale and alternatives considered.

---

## Monitoring

`src/monitoring/drift_monitor.py` runs Evidently AI drift analysis comparing
the training distribution (reference) against incoming production data (current).

Statistical tests:
- Wasserstein distance — continuous features (age, hours_per_week)
- Jensen-Shannon distance — categorical features (marital_status)
- Chi-squared — remaining categorical features

9 metrics published to CloudWatch namespace `ResponsibleRiskEngine/Drift`
on every run. Alert threshold: drift_share > 0.20 triggers retraining
recommendation.

**Virginia baseline:** 0/6 features drifted. Drift share: 0.0.

**Drift Response Workflow:**
When drift_share exceeds 0.20, a CloudWatch alarm fires and notifies
the responsible engineer. The engineer reviews the drift report to
confirm the drift is meaningful before initiating retraining. Automated
retraining without human review is explicitly not implemented — see
DL-015 for full rationale.

---

## Alternative Deployment Targets

The pipeline is platform-agnostic above the serving layer. MLflow, Evidently,
XGBoost, and the fairness audit are not AWS-specific. Replacing SageMaker
requires only a `deploy.py` rewrite — everything else stays unchanged.

### Kubernetes
```
Dockerfile
k8s/deployment.yaml   — pod spec, resource limits, health checks
k8s/service.yaml      — LoadBalancer exposing the inference endpoint
k8s/hpa.yaml          — HorizontalPodAutoscaler for traffic-based scaling
```

The model artifact is pulled from S3 at container startup. The same
joblib-serialized XGBClassifier used locally works inside a standard
Python container.

### Azure Machine Learning
```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint

# Register model in Azure ML Registry
model = Model(path="models/xgboost_20260312.joblib", ...)
client.models.create_or_update(model)

# Deploy to managed online endpoint
endpoint = ManagedOnlineEndpoint(name="responsible-risk-engine")
client.online_endpoints.begin_create_or_update(endpoint)
```

### GCP Vertex AI
```python
from google.cloud import aiplatform

model = aiplatform.Model.upload(
    display_name="income-risk-xgboost",
    artifact_uri="gs://bucket/models/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/..."
)
endpoint = model.deploy(machine_type="n1-standard-4")
```

In all three cases: same training pipeline, same fairness audit, same
MLflow registry, same drift monitoring. Only the deployment target changes.

---

## Production Path

The pipeline is deployed and verified on Virginia (FIPS 51) data.
National expansion is a documented future enhancement:

1. **National data pull** — set `STATE_CODE = "*"` in `config.py`
2. **Retrain** — full pipeline from `ingest.py` through `register.py`
3. **National fairness audit** — Virginia results are not nationally representative
4. **Deployment approval** — human sign-off on national audit results
5. **Monitor** — drift_monitor.py against national training distribution

This is one config change followed by the same pipeline already
proven end-to-end on Virginia data.

---

## Tradeoffs and Future Work

### sklearn Pipeline — Preprocessing Bundling Deferred
The current serving architecture applies preprocessing client-side and
passes preprocessed inputs to the SageMaker endpoint. Bundling preprocessing
and model into a single sklearn Pipeline artifact is the correct long-term
design — it eliminates version mismatch risk and simplifies inference.

This was deferred deliberately. The SageMaker XGBoost managed container
loads native JSON format directly with no custom inference script. A
sklearn Pipeline artifact requires either an sklearn container with a
custom inference script or a custom container with a Dockerfile and ECR
push. Both paths were attempted and failed due to container behavior
documented in DL-011. The working deployment takes priority over
architectural purity at this stage.

**Future work:** Implement full sklearn Pipeline and serve via SageMaker
sklearn container once the container scripting issues are resolved.

### National Scale
The pipeline is proven end-to-end on Virginia data (88,928 records).
National expansion (STATE_CODE="*", ~1.5M records) requires:
- Optuna trial budget review — 30 trials sufficient for Virginia,
  may need adjustment for national feature distributions
- SageMaker endpoint auto-scaling policy — current deployment is
  single instance, appropriate for development verification
- Evidently AI drift analysis memory and runtime review at 1.5M records
- CloudWatch alarm thresholds retuned for national traffic volume
- Full national fairness audit — Virginia results are not nationally
  representative, particularly for small demographic groups

**Future work:** National retrain is one config change. The infrastructure
and monitoring design are already built to support it.

### Retraining Automation
The current retraining trigger is manual — a CloudWatch alarm notifies
an engineer who initiates retraining after human review. EventBridge and
Lambda infrastructure for automated triggering is referenced in the
architecture but retrain_trigger.py is not implemented.

**Future work:** Implement retrain_trigger.py — EventBridge rule fires
on CloudWatch alarm, Lambda initiates the retraining pipeline, human
approval gate remains before production promotion.

### Auto-scaling
The SageMaker endpoint is deployed on a single ml.m5.xlarge instance.
For production traffic, an Application Auto Scaling policy would scale
instance count based on invocation rate and bring instances to zero
during off-peak hours to minimize cost.

**Future work:** Add auto-scaling policy to infrastructure/main.tf.

---

## Repository Structure

```
responsible-mlops-risk-engine/
├── config.py                          # All parameters — single source of truth
├── .env.example                       # Credential template
├── requirements.txt                   # Dependencies — sagemaker pinned to 2.x
├── PORTFOLIO.md                       # Portfolio overview
│
├── src/
│   ├── data/
│   │   ├── ingest.py                  # Census API pull
│   │   └── preprocess.py             # Feature engineering, split
│   ├── training/
│   │   ├── baseline.py               # Logistic Regression
│   │   ├── ridge.py                  # Ridge with L2
│   │   ├── train_xgboost.py          # XGBoost + Optuna
│   │   ├── evaluate.py               # Metrics + fairness gate
│   │   └── register.py               # MLflow registry
│   ├── serving/
│   │   └── deploy.py                 # SageMaker deployment
│   └── monitoring/
│       └── drift_monitor.py          # Evidently AI + CloudWatch
│
├── infrastructure/
│   ├── main.tf                        # S3, IAM, CloudWatch
│   ├── variables.tf
│   └── outputs.tf
│
└── docs/
    ├── decision_log.md               # DL-001 through DL-016
    ├── fairness_report.md            # Stakeholder fairness audit
    ├── nist_alignment.md             # NIST AI RMF 1.0 mapping
    ├── architecture.md               # This document
    ├── model_card.md                 # Model details, intended use, limitations
    └── runbook.md                    # Operational procedures
```

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.9 |
| ML | XGBoost 2.x, Scikit-learn, Optuna |
| Experiment tracking | MLflow 2.x |
| Fairness | Custom evaluate.py — PPR, AUC per demographic group |
| Drift monitoring | Evidently AI 0.7.x |
| Explainability | SHAP (planned — Streamlit demo) |
| Infrastructure | Terraform 1.x, AWS |
| Serving | SageMaker real-time endpoint |
| Storage | S3 (3 buckets — raw, processed, models) |
| Monitoring | CloudWatch, Evidently AI |
| CI/CD | GitHub Actions — flake8, structure validation |
| Demo | Streamlit |
