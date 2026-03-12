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
National pull reserved for final model training.