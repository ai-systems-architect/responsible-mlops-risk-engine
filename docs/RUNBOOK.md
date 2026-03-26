# Runbook
## responsible-mlops-risk-engine

**Date:** 2026-03-26
**Scope:** Operational procedures for deployment, rollback, drift
response, fairness gate failure, and retraining.

---

## Prerequisites

```bash
cd ~/responsible-mlops-risk-engine
source venv/bin/activate
export PYTHONPATH=.

# Verify environment
aws sts get-caller-identity        # confirm AWS credentials
mlflow ui &                        # http://localhost:5000
```

Ensure `.env` is populated:
- `CENSUS_API_KEY`
- `AWS_ACCOUNT_ID`
- `S3_BUCKET`
- `SAGEMAKER_ROLE_ARN`

---

## 1. Full Pipeline Run

Run end-to-end from data ingestion through model registry.

```bash
# Data
python3 src/data/ingest.py
python3 src/data/preprocess.py

# Training
python3 src/training/baseline.py
python3 src/training/ridge.py
python3 src/training/train_xgboost.py

# Evaluation + fairness gate
python3 src/training/evaluate.py
# Exit code 0 = gate passed. Exit code 1 = gate failed — do not proceed.

# Registry
python3 src/training/register.py
```

**Gate check:** `evaluate.py` must exit 0 before proceeding to
deployment. A non-zero exit code means at least one demographic group
exceeded the ±0.20 PPR threshold. See Section 4.

---

## 2. Deployment

**Prerequisites:** Full pipeline run complete, evaluate.py exited 0,
human sign-off obtained on fairness audit results.

```bash
# Infrastructure — provision if not already running
cd infrastructure
terraform init
terraform apply -var="aws_account_id=${AWS_ACCOUNT_ID}"
cd ..

# Deploy endpoint
python3 src/serving/deploy.py
```

**Expected outcome:** SageMaker endpoint reaches `InService` status
within 4–6 minutes. Verify in AWS console or:

```bash
aws sagemaker describe-endpoint \
  --endpoint-name responsible-risk-engine-prod-v1 \
  --query 'EndpointStatus'
```

**Cost reminder:** Endpoint costs ~$5/day. Destroy immediately after
verification if not required for active use — see Section 3.

---

## 3. Endpoint Teardown

```bash
aws sagemaker delete-endpoint \
  --endpoint-name responsible-risk-engine-prod-v1
```

Verify deletion:
```bash
aws sagemaker list-endpoints \
  --query 'Endpoints[?EndpointName==`responsible-risk-engine-prod-v1`]'
```

Terraform-provisioned resources (S3, IAM, CloudWatch) are not affected
by endpoint deletion. To destroy all infrastructure:

```bash
cd infrastructure
terraform destroy -var="aws_account_id=${AWS_ACCOUNT_ID}"
```

---

## 4. Fairness Gate Failure

**Trigger:** `evaluate.py` exits with code 1. CI/CD pipeline blocked.

**Do not override the gate. Do not proceed to deployment.**

**Investigation steps:**

1. Review per-group PPR and AUC output from `evaluate.py`
2. Identify which group(s) exceeded the ±0.20 PPR threshold
3. Determine root cause — options:
   - Data shift in a demographic group (check ingest.py output)
   - Label imbalance introduced during preprocessing
   - Model behavior change from hyperparameter tuning
4. Document findings in `docs/fairness_report.md` before any retraining
5. If root cause is a data issue — fix upstream and re-run from `ingest.py`
6. If root cause is model behavior — re-run `train_xgboost.py` with
   adjusted Optuna trial budget or parameter constraints
7. Re-run `evaluate.py` — gate must pass before proceeding
8. Human sign-off required on new fairness results before deployment

**Gate threshold cannot be modified in CI/CD without a decision log
entry.** Any threshold change requires a new DL entry with explicit
justification. See DL-016.

---

## 5. Drift Alert Response

**Trigger:** CloudWatch alarm fires on `ResponsibleRiskEngine/Drift`
namespace. drift_share metric exceeded 0.20.

**Steps:**

1. Locate the drift report:
```bash
ls -lt drift_report_*.html | head -1
```

2. Open the report and identify which features drifted and by how much

3. Assess whether drift is meaningful:
   - **Data pipeline artifact** — check ingest.py for API changes,
     field mapping changes, or Census Bureau variable updates
   - **Real population shift** — income distribution, occupation mix,
     or demographic composition has changed in the underlying population

4. If drift is a pipeline artifact — fix the pipeline, re-run, verify
   drift clears before proceeding

5. If drift is a real population shift — initiate retraining.
   See Section 6.

6. Document findings and action taken. See DL-015 for full rationale
   on manual vs automated retraining.

---

## 6. Retraining

**Prerequisites:** Drift confirmed as meaningful (Section 5) or
fairness gate failure root cause identified (Section 4).

Retraining follows the same sequence as a full pipeline run —
Section 1. Key considerations:

**Data:** Determine whether new data should be pulled or existing
processed data is sufficient. If population drift is confirmed, a
fresh `ingest.py` run is required.

**Hyperparameters:** Optuna re-runs 30 trials on each retrain.
Parameters from the previous run are not reused — Optuna searches
fresh each time.

**Fairness gate:** Must pass on the new model before promotion
is considered. A model that resolves drift but fails the fairness
gate does not proceed.

**Registry:** `register.py` logs the new model to MLflow as a new
version. The new version receives alias `staging` — it does not
automatically replace the production model.

---

## 7. Production Promotion

**Prerequisites:**
- Full pipeline run complete
- evaluate.py exited 0
- Drift report reviewed (if retrain was drift-triggered)
- Human sign-off obtained

**Promotion steps:**

1. Verify new model version in MLflow registry at `http://localhost:5000`
2. Compare metrics against current production model — AUC, F1,
   per-group fairness results
3. Obtain explicit sign-off from responsible engineer or approver
4. Deploy new model — Section 2
5. Monitor endpoint for first 24 hours — check CloudWatch for error
   rate and latency anomalies
6. Previous model artifact is retained in MLflow registry — not deleted

**No automated promotion path exists by design.** CI/CD gates are
necessary but not sufficient for production promotion. See DL-015.

---

## 8. Local Inference (Streamlit / Batch)

The full sklearn Pipeline artifact (`full_pipeline_*.joblib`) supports
local inference without SageMaker. Use for Streamlit demo and batch
scoring.

```python
import joblib
import pandas as pd

pipeline = joblib.load("models/full_pipeline_YYYYMMDD.joblib")
predictions = pipeline.predict(X_new)
probabilities = pipeline.predict_proba(X_new)[:, 1]
```

For SageMaker endpoint invocation, preprocessing must be applied
client-side before submission. See DL-014.

---

## 9. CI/CD

GitHub Actions runs on every push and pull request to `main`.

**What CI checks:**
- flake8 lint — `src/` and `config.py`
- Project structure validation — config.py, src/data/, docs/decision_log.md

**What CI does not check:**
- Model training — training gate added after register.py is complete
- Fairness audit — runs locally and in full pipeline only
- Deployment — manual step, requires human approval

Pipeline configuration: `.github/workflows/ml-pipeline.yml`
