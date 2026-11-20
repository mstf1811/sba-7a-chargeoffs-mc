# Model note: SBA 7(a) charge-offs (Step 1)

**Notation.** Unit $i=1,\dots,n$ ($n=32{,}806$) is one lender × approval-fiscal-year cohort, FY2000–2019.
$Y_i$ = number of loans charged off; $e_i$ = loan-years at risk (disbursement → charge-off, payoff or the
2026-09-30 cut-off); $x_i$ = covariates (vintage, sector, business type, lender state, loan size, term,
guaranteed share, collateral, Express, young-business shares…); $g(i)$ = lender (4,940 levels);
$X_{ij}$ = gross amount of the $j$-th charge-off of unit $i$. Pooled raw rate: $\sum Y_i / \sum e_i = 30.3$ per 1,000 loan-years.

**Base model.** $\;Y_i \overset{\text{ind.}}{\sim} \text{Poisson}(\lambda e_i)$, with $\lambda$ the charge-off rate per loan-year.

**Why time at risk is the right exposure.** If each loan has a constant hazard $\lambda$ and is censored at
payoff or cut-off, its likelihood is $\lambda^{\delta}e^{-\lambda t}$; over a unit this gives
$\lambda^{Y_i}e^{-\lambda e_i}$, which is the Poisson$(\lambda e_i)$ likelihood up to a factor free of $\lambda$.
The Poisson model with offset $\log e_i$ is therefore a censored-exponential survival model: same MLE,
same tests, same posterior.

**Sufficiency.** $L(\lambda)=\lambda^{T}e^{-\lambda E}\prod_i e_i^{y_i}/y_i!$ with $T=\sum_i Y_i$, $E=\sum_i e_i$:
by factorisation $T$ is sufficient. Equivalently $Y \mid T=t \sim$ Multinomial$(t;\,e_i/E)$, free of $\lambda$
(checked by Monte Carlo in notebook 01, seed 20261120).

## Assumptions and how each will be tested

| Assumption | Doubtful here? | Evidence (notebook 01) | Test |
|---|---|---|---|
| **A1 Independence** across units | **Yes.** All cohorts of a vintage face the same economy; one lender's cohorts share its underwriting | FY2005–08: 45–69 per 1,000 vs 11–16 for FY2011–19 | Residuals by year and robust SE clustered by year (step 9); cohort-year frailty vs independent model for the loss tail (step 11) |
| **A2 Homogeneity**: one λ for all | **Yes, false.** | Rate ×6 across vintages, ×5 across sectors, ×4 across loan terms, far outside Poisson intervals | NB vs Poisson (step 2); LRT on the boundary and lender-vs-overall tests with Bonferroni / BH (step 4); covariate blocks in the GLM (step 9) |
| **A3 Proportionality to e** (constant hazard) | **Partly.** Charge-offs peak 2–5 years after disbursement, and e ends at the charge-off | Rate flat (17–20) over exposure deciles 3–9 but 67 and 32 at the ends; Poisson slope of $\log e$ = 1.19 | Free coefficient on $\log e$ in the GLM, Wald / LR test of slope = 1 (step 9) |
| **A4 Equidispersion** Var = mean | **Yes**, a consequence of A2 failing | Not tested yet | Moments and MLE NB fits (step 2); boundary LRT and its parametric bootstrap (step 4); dispersion statistic, quasi-Poisson, sandwich SE, NB GLM (step 9) |
| **A5 Exposure without error** | **Yes (data).** `EXEMPT` loans are counted at risk until the cut-off; charge-off dates lag defaults | 49,926 `EXEMPT` loans, 48 % of charge-offs dated after maturity | Sensitivity refit without `EXEMPT` loans (step 9) |

**Reading.** The base model is a benchmark, not a credible description: homogeneity and independence fail
for structural reasons (vintage effects and the 2008–09 recession). The project moves away from it in three
directions: extra-Poisson variance (negative binomial, steps 2–4), lender-level shrinkage (step 5), and
covariates (GLM and machine learning, steps 6–10). The time dependence is handled in the loss tail (step 11).
