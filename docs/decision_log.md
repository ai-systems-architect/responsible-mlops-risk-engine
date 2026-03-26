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
National pull is a documented future enhancement — see DL-012.

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
## DL-010 — SageMaker Python SDK Pinned to 2.x
**Date:** 2026-03-19
**Decision:** SageMaker SDK constrained to `>=2.200.0,<3.0.0` in requirements.txt
**Rationale:** SageMaker 3.x (released early 2026) is a complete API rewrite.
All framework-specific classes — XGBoostModel, SKLearnModel, XGBoostPredictor —
were removed and replaced with unified ModelBuilder and ModelTrainer classes.
Version 3.x is too new and poorly documented for production use. Version 2.257.1
is the stable, well-documented version used across all production pipelines.

---

## DL-011 — Native XGBoost Format for SageMaker Deployment
**Date:** 2026-03-19
**Decision:** Model saved in XGBoost native JSON format for SageMaker deployment
**Rationale:** The joblib-serialized XGBClassifier (sklearn API) repeatedly failed
in the SageMaker XGBoost container due to container behavior — the container always
runs `pip install .` on any code directory it finds, then attempts to import the
installed package. This caused import failures regardless of script naming.

The native XGBoost JSON format eliminates the need for a custom inference script
entirely. The container loads the model directly with no pip install, no import
issues, and no script management. The joblib artifact is retained locally for
MLflow logging and Streamlit inference.

Conversion:
```python
model.get_booster().save_model('models/xgboost_native.json')
```

**Result:** Endpoint reached InService in 4 minutes. All predictions consistent
with local model — delta < 0.001 across all test records.

---

## DL-012 — National Data Pull Deferred to Production
**Date:** 2026-03-19
**Decision:** Pipeline developed and deployed on Virginia (FIPS 51) data only.
National retrain deferred.
**Rationale:** Virginia data (88,928 records) provides sufficient volume for
development iteration and model validation. The full pipeline — ingestion,
preprocessing, training, fairness audit, MLflow registry, and SageMaker deployment
— is proven end-to-end on Virginia data.

National retrain requires one configuration change:
```python
STATE_CODE = "*"  # config.py — pulls ~1.5M records across all 50 states
```

This is the documented production path. Federal pilots always start with a state
before going national — this approach reflects real government delivery practice.

---
## DL-014 — SageMaker Endpoint Receives Preprocessed Inputs
**Date:** 2026-03-20
**Decision:** SageMaker endpoint accepts preprocessed inputs. Full sklearn Pipeline
used client-side (app.py, batch scoring) rather than inside the container.
**Rationale:** The SageMaker XGBoost managed container can only serve XGBoost's
native binary/JSON format. It cannot load or run a full sklearn Pipeline object.

The serving architecture is:

```
Client (app.py / API caller)
    ↓  raw inputs (age, education codes, occupation codes)
    ↓  pipeline.named_steps["preprocessor"].transform()
    ↓  preprocessed CSV
SageMaker Endpoint
    ↓  native XGBoost inference
    ↓  probability score
Client
```

The full sklearn Pipeline (`full_pipeline_*.joblib`) is the primary artifact
for local inference, batch scoring, and the Streamlit demo fallback. The
SageMaker endpoint is the production serving layer for real-time API calls.

This is the standard pattern for XGBoost on SageMaker. Alternatives considered:

    1. sklearn container — can serve full pipeline but requires custom inference
       script, which caused repeated container failures (see DL-011).
    2. Custom container — full control but requires Dockerfile and ECR push,
       adding infrastructure complexity not justified at this scope.
    3. Current approach — clean separation of concerns, proven working deployment.

---

---
## DL-015 — Drift Response Workflow
**Date:** 2026-03-26
**Decision:** Drift response is manual review triggered by CloudWatch alarm,
not automated retraining.
**Rationale:** Evidently AI drift_monitor.py publishes 9 metrics to CloudWatch
namespace ResponsibleRiskEngine/Drift on every run. When drift_share exceeds
0.20, a CloudWatch alarm fires. The response workflow is:

1. CloudWatch alarm notifies the responsible engineer via SNS
2. Engineer reviews the drift report (drift_report_*.html) to identify
   which features drifted and by how much
3. If drift is confirmed as meaningful — not a data pipeline artifact —
   a retraining run is initiated manually
4. Full pipeline runs from ingest.py through register.py
5. Fairness gate must pass before the new model is eligible for promotion
6. Human sign-off required before promoting new model to production
7. Previous model remains live until new model is approved and deployed

**Rationale for manual retraining over automated:** Automated retraining
without human review risks promoting a model that passes AUC and fairness
gates on drifted data that no longer reflects the intended population.
In a government-aligned pipeline, human review before promotion is a
requirement, not a preference.

---
## DL-016 — Fairness Gate Failure Recovery
**Date:** 2026-03-26
**Decision:** Fairness gate failure blocks deployment and requires documented
human investigation before retraining is attempted.
**Rationale:** evaluate.py exits with code 1 when any demographic group
exceeds ±0.20 PPR threshold, blocking all downstream steps. Recovery process:

1. Engineer reviews per-group PPR and AUC results in the run output
2. Root cause investigation — determine whether failure is caused by:
   - Data shift in a demographic group
   - Label imbalance introduced by preprocessing
   - Model behavior change from hyperparameter tuning
3. Findings documented in docs/fairness_report.md before any retraining
4. Retraining initiated only after root cause is identified and documented
5. Human sign-off required on fairness results before production promotion
6. Gate threshold cannot be overridden in CI/CD — any override requires
   a decision log entry with explicit justification

**No silent override path exists by design.** A failing fairness gate
that proceeds to deployment without documented human review is an audit
failure in a government-delivery context.
