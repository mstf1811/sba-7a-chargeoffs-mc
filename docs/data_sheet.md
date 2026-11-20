# Data sheet — SBA 7(a) charge-offs by lender cohort

**Guiding question.** For the U.S. SBA 7(a) small-business loan book: how many loans will charge off next year, what gross credit loss does that imply, what is the 99.5 % (and Basel 99.9 %) quantile of that loss, and does empirical-Bayes shrinkage change which lenders a regulator would flag as high-risk?

## Source

| | |
|---|---|
| Publisher | U.S. Small Business Administration, Office of Capital Access |
| Dataset | "7(a) & 504 FOIA", https://data.sba.gov/dataset/7a-504-foia |
| Files used | `FOIA_7a_FY2000_FY2009_asof_260930.csv`, `FOIA_7a_FY2010_FY2019_asof_260930.csv`, `FOIA_7a_FY2020_Present_asof_260930.csv`, `7a_504_foia_data_dictionary.xlsx` (direct links under `https://data.sba.gov/sites/default/files/uploaded_resources/`) |
| Licence | U.S. Government Works (public domain in the U.S.), so teaching use is allowed |
| Snapshot / update | Data as of 2026-09-30, published 2026-10-30, updated quarterly |
| Download date | 2026-11-17 (byte size + SHA-256 of each file in `data/raw/manifest.json`) |
| Loading script | `python scripts/build_data.py`, which calls `sba_mc.download` and then `sba_mc.data.load_data()` |

## Size

1,639,728 raw loan records (FY2000 to present) give 1,410,206 disbursed loans after cleaning. The analysis table keeps **FY2000–FY2019**: **1,081,254 loans** grouped into **32,806 units** from **4,940 lenders**, with **178,803 charge-offs** over **5.90 M loan-years** at risk (pooled rate 30.3 per 1,000 loan-years). FY2020+ cohorts are held out as an out-of-time check because they are too right-censored to model.

## Unit and role mapping

**Unit i = lender (`LocationID`) × approval fiscal-year cohort.**

| Role | Column | Construction |
|---|---|---|
| Count Y_i | `Y` | Number of loans in the cohort with status `CHGOFF`. 63 % of units have Y = 0, median 0, max 4,983 |
| Exposure e_i | `e` | Loan-years at risk: Σ (end − start)/365.25. Start = first disbursement (approval date if missing). End = charge-off date, paid-in-full date, or the 2026-09-30 cut-off. Median 17.3, max 49,956 |
| Continuous covariates | `mean_log_amount`, `mean_term_months`, `guar_share`, `mean_jobs`, `share_express`, `share_young`, `share_collateral`, `share_revolver`, `share_franchise`, `n_loans` | Cohort means or shares of loan-level fields |
| Categorical covariates | `fy` (20 levels), `sector` (NAICS 2-digit, dominant, 20 levels), `business_type` (3+1), `lender_state` (61) | Mode within the cohort |
| Groups (steps 4–5) | `lender` | 4,940 levels, very unequal: median 7 loans per lender, 2,693 lenders with fewer than 10 loans, largest 73,342 |
| Amounts (step 11) | `load_amounts()` → `chgoff_amount` | `GrossChargeOffAmount` for each charged-off loan, linked to its (lender, fy) unit. Median $40.9 k, mean $109 k, skewness 5.4 |

**Why this exposure works.** If each loan has a constant hazard λ, the likelihood of its (event, time-at-risk) is the censored exponential λ^δ e^{−λt}. Summed over a unit, this is proportional to the Poisson(λ·e_i) likelihood of Y_i, so a Poisson GLM with offset log e_i is exactly a piecewise-constant survival model.

## Cleaning, quantified

| Step | Rows |
|---|---|
| Exact duplicates removed | −2,383 |
| `CANCLD` / `COMMIT` (never disbursed, no risk) | −221,847 |
| Missing lender ID or FY | −14 |
| Non-positive time at risk (all `PIF` with a payoff date on or before disbursement) | −5,278 |
| Disbursement date missing, approval date used instead | 1,545 imputed |
| `CHGOFF` / `PIF` with missing end date, censored at cut-off | 26 |

## Known quality issues

1. **Charge-off date lags default.** 48 % of charge-offs are dated after scheduled maturity. Exposure is overstated and timing is blurred, so we model the *charge-off* event, not the default.
2. **Amounts are gross**, not net of recoveries, and include guaranteed and unguaranteed parts. 18 are ≤ 0 and get dropped in step 11. Values cluster at round numbers ($50 k appears 2,076 times).
3. **`EXEMPT` status.** Under FOIA Exemption 4, the status of disbursed loans that are still open is hidden. 49,926 FY2000–19 loans are treated as at risk until the cut-off, and 6.9 % of them are more than 2 years past maturity, which probably overstates e.
4. **Non-constant hazard and right-censoring.** Charge-offs peak about 2–5 years after disbursement, so recent cohorts look safer partly because of the time window. `fy` is kept as a covariate, and FY2020+ is excluded.
5. **Dependence.** Defaults cluster by vintage: 68.7 per 1,000 loan-years for FY2007 vs 11.1 for FY2013. This violates independence across units in the same year, and we address it with a cohort-year frailty in step 11.
6. **Lender = current servicer.** `LocationID` / `BankName` is the lender the loan is *currently* assigned to, so mergers move loans between lenders.
7. **Fields not used.** `InitialInterestRate` is missing for 100 % of loans approved before FY2009. `SoldSecMrktInd` is blank for 72 %. NAICS is missing for early loans (sector "Unknown", 931 units). `JobsSupported` is self-reported by lenders. The dictionary codes `RevolverStatus` as 0/1, but the files use N/Y.
8. **Moving target.** Files are replaced quarterly, so we pin the `asof_260930` file names and check them against the stored SHA-256.
