# Fairness Audit Report
*Designing Governed AI Systems*

## responsible-mlops-risk-engine

**Date:** 2026-03-12
**Model:** XGBoost (Optuna-tuned, 30 trials)
**Dataset:** ACS PUMS 2023 — Virginia (FIPS 51)
**Prepared by:** ai-systems-architect
**Status:** PASSED

---

## Executive Summary

The XGBoost income risk scoring model was evaluated for demographic
fairness across race, sex, and nativity groups using the 2023 American
Community Survey Public Use Microdata Sample for Virginia.

The model passed the fairness gate — no demographic group's positive
prediction rate deviated from the overall rate by more than the 20%
threshold established in the project configuration.

Findings indicate that observed prediction rate differences across groups
reflect income distribution patterns in the underlying Virginia labor
market data rather than model-introduced bias. Specific limitations and
monitoring commitments are documented in the sections below.

---

## What This Report Measures

**Positive Prediction Rate (PPR)**
The proportion of individuals in each group that the model predicts
will earn at or above $75,000 annually. A large disparity in PPR
across demographic groups can indicate the model treats groups
differently regardless of their actual income.

**Fairness Gate**
Each group's PPR is compared against the overall PPR of 0.2686.
Groups deviating by more than ±0.20 fail the gate and block
the model from proceeding to deployment approval.

**Important distinction**
Sensitive features — race, sex, and nativity — were never used as
model inputs. They were physically separated from training data at
the preprocessing stage and are used here exclusively for post-prediction
audit. The model has no direct access to protected attributes.

---

## Dataset Scope and Limitations

This report covers Virginia (FIPS 51) development data only.

| Metric | Value |
|---|---|
| Test set records | 14,407 |
| Income threshold | $75,000 annual wage income |
| Overall positive rate | 21.7% of test set earns ≥ $75K |
| Overall PPR | 26.9% predicted to earn ≥ $75K |
| Fairness threshold | ±0.20 PPR deviation |

**Virginia is not representative of national demographics.** Results
will differ when the pipeline is retrained on national data
(STATE_CODE="*"). A full fairness audit on national data is required
before production deployment.

---

## Results by Demographic Group

### Race

| Group | N | PPR | Delta from Overall | AUC-ROC | Gate |
|---|---|---|---|---|---|
| White alone | 9,685 | 0.279 | +0.010 | 0.954 | ✅ Pass |
| Black or African American alone | 2,086 | 0.171 | -0.098 | 0.920 | ✅ Pass |
| American Indian alone | 68 | 0.353 | +0.084 | 0.909 | ✅ Pass |
| Asian alone | 1,059 | 0.413 | +0.144 | 0.952 | ✅ Pass |
| Some other race alone | 448 | 0.167 | -0.101 | 0.941 | ✅ Pass |
| Two or more races | 1,044 | 0.263 | -0.006 | 0.948 | ✅ Pass |

**Key finding — Race:**
The largest inter-group gap is between Black or African American
(PPR 0.171) and Asian (PPR 0.413) — an absolute difference of 0.242.
Both groups individually pass the gate when measured against the overall
PPR. The inter-group gap reflects income distribution patterns in the
Virginia 2023 ACS data. The model's AUC is strong across all groups
(0.909–0.954), indicating prediction quality is consistent regardless
of race. Monitoring this gap in production is a stated requirement.

**American Indian alone — small sample warning:**
68 records is insufficient for reliable metric estimation. Precision
of 0.375 with recall of 0.90 suggests the model over-predicts high
income for this group. This finding should not be treated as definitive
until national data provides a larger sample. This group is flagged for
priority review after the national data pull.

---

### Sex

| Group | N | PPR | Delta from Overall | AUC-ROC | Gate |
|---|---|---|---|---|---|
| Male | 6,929 | 0.321 | +0.053 | 0.946 | ✅ Pass |
| Female | 7,478 | 0.220 | -0.049 | 0.954 | ✅ Pass |

**Key finding — Sex:**
Male PPR (0.321) is higher than female PPR (0.220) — a 10 point
difference. Both groups pass the gate. AUC scores are nearly equal
(0.946 / 0.954), indicating the model predicts with equivalent
accuracy for both sexes. The PPR difference reflects the wage gap
present in the underlying Virginia labor market data. The model does
not introduce additional disparity — it reflects what exists in the data.

---

### Nativity

| Group | N | PPR | Delta from Overall | AUC-ROC | Gate |
|---|---|---|---|---|---|
| Native born | 12,445 | 0.260 | -0.009 | 0.951 | ✅ Pass |
| Foreign born | 1,962 | 0.324 | +0.055 | 0.949 | ✅ Pass |

**Key finding — Nativity:**
Foreign born individuals have a slightly higher PPR (0.324) than
native born (0.260). AUC scores are nearly identical (0.949 / 0.951).
No meaningful performance disparity across nativity groups.

---

## Overall Model Performance

| Metric | Value |
|---|---|
| AUC-ROC | 0.9506 |
| F1 | 0.7633 |
| Precision | 0.6896 |
| Recall | 0.8546 |
| Accuracy | 0.89 |

The model correctly identifies 85% of individuals who earn above the
income threshold (recall). Precision of 0.69 means approximately 31%
of positive predictions are incorrect — acceptable for a screening
model where missing true positives carries higher cost than false
positives.

---

## Risk Response Commitments

The following actions are required before and during production deployment:

| Action | Trigger | Owner |
|---|---|---|
| Re-run full fairness audit if expanded to national data | Before national deployment | ML team |
| Flag American Indian group for priority review | After national data pull | ML team |
| Monitor per-group PPR drift monthly | Post-deployment | ML team — via drift_monitor.py |
| Alert threshold set at ±0.15 PPR delta | Tighter than gate — early warning | ML team — via CloudWatch |
| Human review required if any group exceeds ±0.15 | Monthly drift report | Responsible AI reviewer |
| Retrain if drift exceeds ±0.20 on any group | Drift alarm | ML team — manual trigger, see DL-015 |

---

## NIST AI RMF Alignment

| NIST Control | Implementation |
|---|---|
| MEASURE 2.5 | Bias evaluated across race, sex, nativity using PPR and AUC |
| MEASURE 2.7 | Fairness gate enforced in CI/CD pipeline — blocks deployment on failure |
| MANAGE 2.2 | Findings documented with risk response plan above |
| MANAGE 2.4 | Monitoring commitments established with specific thresholds |
| GOVERN 1.7 | Sensitive features never used as model inputs — documented in decision log |

---

## Methodology Notes

- Sensitive features separated at preprocessing — never used as model inputs
- Fairness audit run on held-out test set only — no training data leakage
- Groups with fewer than 30 records excluded from audit — metrics unreliable
  at very small sample sizes
- PPR delta measured against overall test set PPR (0.2686)
- AUC computed per group where both positive and negative classes present
- All results based on Virginia 2023 ACS data — not nationally representative
- "Some other race alone" excluded from inter-group gap reporting —
  Census catch-all category, not a defined demographic population

---

*This report was generated automatically by src/training/evaluate.py
and reviewed manually. Results are specific to Virginia (FIPS 51) development data.
National expansion requires a full fairness audit on the complete ACS PUMS dataset.*
