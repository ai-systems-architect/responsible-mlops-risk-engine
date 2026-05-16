# ADR-0001 — Inference Serving Pattern and Rollback Strategy

**Status:** Target Design (canary/blue-green deferred to production hardening per DL-019)
**Date:** 2026-05-16
**Deciders:** Raghu Devayajanam
**Related:** DL-019, DL-020, architecture.md § Deployment Patterns, runbook.md §7

---

## Context

The income risk scoring system (IRSS) requires a serving layer for
synchronous decision-support scoring — a screening or eligibility worker
submits applicant data and needs a probability back in the moment.
Three SageMaker serving patterns were evaluated at project outset. A
rollback strategy is required because model promotion is governed: any
degradation in a new model version must have a defined recovery path,
and the recovery path itself is part of the architectural decision.

---

## Options Evaluated

| Pattern | Latency | Cost Model | Use Case Fit | IRSS Verdict |
|---|---|---|---|---|
| Real-time endpoint | Sub-second | ~$5.50/day on ml.m5.xlarge, always-on | Synchronous decision-support, live requests | **Selected** — synchronous scoring requires real-time response |
| Batch transform | Minutes to hours | Pay-per-job, no idle cost | Periodic bulk scoring, offline analytics | **Deferred** — appropriate for the future national re-scoring phase (STATE_CODE='*'), not interactive use |
| Serverless inference | Sub-second warm, 1–5s cold start | Pay-per-invocation | Low-frequency workloads where cold start is acceptable | **Rejected** — cold start unacceptable for live screening requests |

---

## Current State

Single production variant deployed via `model.deploy()` in
`src/serving/deploy.py` on ml.m5.xlarge (~$5.50/day when active).
Rollback is in-place redeployment against a previous MLflow model URI —
approximately 6 minutes of endpoint unavailability per runbook §7.

CloudWatch alarms active per `infrastructure/main.tf`:
- `invocation_errors` — threshold 5% of requests
- `model_latency` — threshold p99 2000ms

Deployment approval gate: four stages per `docs/nist_alignment.md` § MANAGE 1.3 —
1. `evaluate.py` — AUC and fairness gate (automated)
2. MLflow registry staging alias (manual)
3. `deploy.py` — manual execution (no automated promotion)
4. SageMaker endpoint pause-before-destroy confirmation (manual)

---

## Target Design

Two production variants deployed simultaneously with SageMaker native
traffic splitting. Canary pattern for model promotion:

1. Register new model version in MLflow — human sign-off required
2. Deploy new version as a second production variant at 10% traffic
3. Monitor for 24 hours:
   - `invocation_errors` CloudWatch alarm stays below threshold
   - `model_latency` p99 CloudWatch alarm stays below threshold
   - Manual fairness check on canary prediction sample — procedure to be
     added to runbook §7.x when canary infrastructure is built
4. On pass: shift traffic to 100% new variant
5. On any fail signal: traffic shift back to previous variant — no
   redeployment, no unavailability window

Rollback becomes a traffic-weight update on the existing endpoint
configuration. The previous model variant remains deployed throughout
the canary soak window for exactly this reason: instant rollback, no
redeployment lag.

**Implementation requires:**
1. `production_variant` blocks in `deploy.py` with `VariantWeight`
   parameters — currently `model.deploy()` registers a single variant
2. `aws_sagemaker_endpoint` Terraform resource with traffic weights —
   currently the endpoint is managed via the SageMaker SDK, not IaC

---

## Consequences

- Current rollback incurs ~6 minutes of endpoint unavailability. Target
  design eliminates this via traffic shift on the existing endpoint
  configuration.
- Canary requires `deploy.py` variant registration and an
  `aws_sagemaker_endpoint` Terraform resource — **not yet implemented**.
- Manual fairness check on canary sample is part of the target design
  and requires a new runbook §7.x sub-procedure when canary
  infrastructure is built.
- Batch re-scoring capability is deferred — the appropriate pattern for
  the national deployment phase, not in current scope.
- Two production variants deployed simultaneously implies a brief cost
  increase during the canary promotion window — acceptable for the risk
  reduction it provides.

---

## NIST AI RMF Alignment

| Control | Current Implementation | Target Design |
|---|---|---|
| MANAGE 1.3 | Four-stage deployment approval gate per `docs/nist_alignment.md` | Canary adds a fifth structural gate — traffic split with soak window before full promotion |
| MANAGE 2.4 | CloudWatch alarms active — `invocation_errors` and `model_latency` per `infrastructure/main.tf` | Same alarms serve as canary pass/fail criteria during the soak window |
| MANAGE 4.1 | Rollback to previous MLflow-registered model URI, ~6 min unavailability | Rollback becomes traffic-weight update — zero unavailability |
| GOVERN 1.4 | MLflow registry tracks all model versions; previous version retained through rollback window | Same — previous variant remains deployed throughout canary soak period |
