import numpy as np
import pandas as pd

from sba_mc.data import build_units, clean_loans
from sba_mc.simulate import mc_mle_poisson, mc_se_proportion, summarize_mc


def test_mle_poisson_unbiased_and_attains_crb():
    rng = np.random.default_rng(0)
    expo = rng.gamma(2.0, 3.0, size=2_000)
    lam = 0.02
    lh, crb = mc_mle_poisson(lam, expo, n=1_000, R=4_000, rng=rng)
    s = summarize_mc(lh, lam)
    assert abs(s["bias"]) < 4 * s["mc_se_bias"]
    # E[var(lam_hat | e)] = E[lam / sum e]: empirical variance should match within ~10%
    assert abs(s["variance"] / crb.mean() - 1) < 0.1


def test_mc_se_proportion():
    assert np.isclose(mc_se_proportion(0.95, 1_000), np.sqrt(0.95 * 0.05 / 1_000))


def _toy_raw():
    return pd.DataFrame({
        "LocationID": ["A", "A", "A", "B", "B"],
        "BankName": ["Bk A"] * 3 + ["Bk B"] * 2,
        "BankState": ["CA"] * 3 + ["NY"] * 2,
        "GrossApproval": ["100000"] * 5,
        "SBAGuaranteedApproval": ["75000"] * 5,
        "ApprovalDate": ["2010-01-01"] * 5,
        "ApprovalFY": ["2010"] * 5,
        "FirstDisbursementDate": ["2010-01-01", "2010-01-01", "", "2010-01-01", "2010-01-01"],
        "ProcessingMethod": ["7a General"] * 5,
        "InitialInterestRate": ["6"] * 5,
        "FixedorVariableInterestInd": ["V"] * 5,
        "TermInMonths": ["120"] * 5,
        "NaicsCode": ["722511", "722511", "445110", "236220", ""],
        "FranchiseCode": ["", "", "", "", ""],
        "ProjectState": ["CA"] * 3 + ["NY"] * 2,
        "BusinessType": ["CORPORATION"] * 5,
        "BusinessAge": [""] * 5,
        "LoanStatus": ["CHGOFF", "P I F", "EXEMPT", "CANCLD", "CHGOFF"],
        "PaidInFullDate": ["", "2012-01-01", "", "", ""],
        "ChargeOffDate": ["2011-01-01", "", "", "", "2014-01-01"],
        "GrossChargeOffAmount": ["50000", "0", "0", "0", "20000"],
        "RevolverStatus": ["N", "Y", "N", "N", "N"],
        "JobsSupported": ["5"] * 5,
        "CollateralInd": ["Y"] * 5,
        "SoldSecMrktInd": [""] * 5,
    })


def test_clean_and_aggregate_toy():
    loans, log = clean_loans(_toy_raw(), cutoff=pd.Timestamp("2020-01-01"))
    assert len(loans) == 4                      # CANCLD dropped
    assert loans["chgoff"].sum() == 2
    a = loans[loans["LocationID"] == "A"].set_index("LoanStatus")["years_at_risk"]
    assert np.isclose(a["CHGOFF"], 1.0, atol=0.01)
    assert np.isclose(a["PIF"], 2.0, atol=0.01)
    assert np.isclose(a["EXEMPT"], 10.0, atol=0.01)   # censored at cut-off
    units = build_units(loans, 2000, 2019)
    assert set(units["lender"]) == {"A", "B"}
    ua = units.set_index("lender").loc["A"]
    assert ua["Y"] == 1 and ua["n_loans"] == 3
    assert np.isclose(ua["e"], 13.0, atol=0.02)
    assert ua["sector"] == "AccommodationFood"
    assert np.isclose(ua["share_revolver"], 1 / 3)
