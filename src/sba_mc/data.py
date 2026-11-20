"""Load, clean and aggregate the SBA 7(a) FOIA files.

Two levels are produced:

* loan level (``load_loans``): one row per disbursed 7(a) loan, with its
  time at risk (``years_at_risk``) and an event flag (``chgoff``).
* unit level (``load_data``): one row per lender x approval-fiscal-year cohort,
  with ``Y`` = number of charge-offs and ``e`` = loan-years at risk. This is the
  table the project's 12 steps work on.

Exposure. A loan is at risk from first disbursement (approval date if missing)
until the first of: charge-off date, paid-in-full date, data cut-off. With a
constant hazard lambda, the Poisson likelihood of (Y, e) with offset log(e) is
proportional to the censored exponential survival likelihood of the loans in
the unit, which is why the Poisson model on these units is meaningful.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from .download import RAW_DIR, ROOT, SNAPSHOT

PROCESSED_DIR = ROOT / "data" / "processed"
CUTOFF = pd.Timestamp("2026-09-30")  # AsOfDate of the snapshot

DECADE_FILES = {
    "2000s": f"FOIA_7a_FY2000_FY2009_asof_{SNAPSHOT}.csv",
    "2010s": f"FOIA_7a_FY2010_FY2019_asof_{SNAPSHOT}.csv",
    "2020s": f"FOIA_7a_FY2020_Present_asof_{SNAPSHOT}.csv",
    "1990s": f"FOIA_7a_FY1991_FY1999_asof_{SNAPSHOT}.csv",
}

USECOLS = [
    "LocationID", "BankName", "BankState", "GrossApproval", "SBAGuaranteedApproval",
    "ApprovalDate", "ApprovalFY", "FirstDisbursementDate", "ProcessingMethod",
    "InitialInterestRate", "FixedorVariableInterestInd", "TermInMonths", "NaicsCode",
    "FranchiseCode", "ProjectState", "BusinessType", "BusinessAge", "LoanStatus",
    "PaidInFullDate", "ChargeOffDate", "GrossChargeOffAmount", "RevolverStatus",
    "JobsSupported", "CollateralInd", "SoldSecMrktInd",
]
DATE_COLS = ["ApprovalDate", "FirstDisbursementDate", "PaidInFullDate", "ChargeOffDate"]

# 2-digit NAICS -> sector (31-33, 44-45, 48-49 are merged as in the official scheme)
NAICS_SECTOR = {
    "11": "Agriculture", "21": "Mining", "22": "Utilities", "23": "Construction",
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",
    "42": "Wholesale", "44": "Retail", "45": "Retail", "48": "Transport", "49": "Transport",
    "51": "Information", "52": "Finance", "53": "RealEstate", "54": "Professional",
    "55": "Management", "56": "AdminSupport", "61": "Education", "62": "HealthCare",
    "71": "Arts", "72": "AccommodationFood", "81": "OtherServices", "92": "PublicAdmin",
}


# --------------------------------------------------------------------------- #
# Loan level
# --------------------------------------------------------------------------- #
def read_raw(decades=("2000s", "2010s", "2020s"), raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read the raw FOIA CSVs (selected columns only) and concatenate them."""
    frames = []
    for d in decades:
        path = raw_dir / DECADE_FILES[d]
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `python -m sba_mc.download` first")
        # read every column as text so IDs keep their leading zeros ("0024910")
        opts = pacsv.ConvertOptions(include_columns=USECOLS,
                                    column_types={c: pa.string() for c in USECOLS})
        df = pacsv.read_csv(path, convert_options=opts).to_pandas()
        df["source_file"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean_loans(raw: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Type, filter and derive exposure. Returns (loans, cleaning_log)."""
    df = raw.copy()
    log = [("raw rows", len(df))]

    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].str.strip()
            df[c] = s.mask(s.isin(["", "None", "nan"]))  # the FOIA files write missing as "None"
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce")
    for c in ["GrossApproval", "SBAGuaranteedApproval", "InitialInterestRate", "TermInMonths",
              "GrossChargeOffAmount", "JobsSupported"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ApprovalFY"] = pd.to_numeric(df["ApprovalFY"], errors="coerce").astype("Int64")
    df["LoanStatus"] = df["LoanStatus"].str.replace(" ", "", regex=False)  # "P I F" -> "PIF"

    # exact duplicates
    n0 = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "source_file"])
    log.append(("drop exact duplicates", len(df) - n0))

    # never disbursed / cancelled loans carry no risk
    n0 = len(df)
    df = df[~df["LoanStatus"].isin(["CANCLD", "COMMIT"])]
    log.append(("drop CANCLD / COMMIT", len(df) - n0))

    n0 = len(df)
    df = df[df["LocationID"].notna() & df["ApprovalFY"].notna()]
    log.append(("drop missing lender or FY", len(df) - n0))

    # exposure window
    start = df["FirstDisbursementDate"].fillna(df["ApprovalDate"])
    df["start_imputed"] = df["FirstDisbursementDate"].isna()
    end = pd.Series(cutoff, index=df.index)
    is_co = df["LoanStatus"].eq("CHGOFF")
    is_pif = df["LoanStatus"].eq("PIF")
    end = end.where(~is_co, df["ChargeOffDate"])
    end = end.where(~is_pif, df["PaidInFullDate"])
    df["end_missing"] = (is_co | is_pif) & end.isna()
    end = end.fillna(cutoff).clip(upper=cutoff)
    df["start"], df["end"] = start, end
    df["years_at_risk"] = (end - start).dt.days / 365.25

    n0 = len(df)
    bad = df["years_at_risk"].isna() | (df["years_at_risk"] <= 0)
    df = df[~bad]
    log.append(("drop non-positive / undefined time at risk", len(df) - n0))

    df["chgoff"] = is_co.loc[df.index].astype(np.int8)
    df["chgoff_amount"] = df["GrossChargeOffAmount"].where(df["chgoff"] == 1)

    # derived covariates
    df["guar_share"] = (df["SBAGuaranteedApproval"] / df["GrossApproval"]).clip(0, 1)
    naics2 = df["NaicsCode"].astype(str).str[:2]
    df["sector"] = naics2.map(NAICS_SECTOR).fillna("Unknown")
    df["BusinessType"] = df["BusinessType"].fillna("UNKNOWN")
    # dictionary says 0/1 but the files use N/Y
    df["revolver"] = df["RevolverStatus"].isin(["1", "Y"]).astype(np.int8)
    df["express"] = df["ProcessingMethod"].fillna("").str.contains("Express").astype(np.int8)
    df["young_business"] = df["BusinessAge"].fillna("").str.contains(
        "Startup|New|2 years or less", regex=True).astype(np.int8)
    df["collateral"] = (df["CollateralInd"] == "Y").astype(np.int8)
    df["sold_secmkt"] = (df["SoldSecMrktInd"] == "Y").astype(np.int8)
    df["franchise"] = df["FranchiseCode"].notna().astype(np.int8)
    df["fixed_rate"] = (df["FixedorVariableInterestInd"] == "F").astype(np.int8)

    log.append(("final loans", len(df)))
    return df, pd.DataFrame(log, columns=["step", "rows"])


def load_loans(decades=("2000s", "2010s", "2020s"), use_cache: bool = True) -> pd.DataFrame:
    """Loan-level table (cached as parquet in data/processed)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache = PROCESSED_DIR / f"loans_{'_'.join(decades)}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    loans, log = clean_loans(read_raw(decades))
    loans.to_parquet(cache, index=False)
    log.to_csv(PROCESSED_DIR / f"cleaning_log_{'_'.join(decades)}.csv", index=False)
    return loans


# --------------------------------------------------------------------------- #
# Unit level: lender x approval-FY cohort
# --------------------------------------------------------------------------- #
def _mode(s: pd.Series):
    m = s.mode()
    return m.iloc[0] if len(m) else np.nan


def build_units(loans: pd.DataFrame, fy_min: int = 2000, fy_max: int = 2019) -> pd.DataFrame:
    """Aggregate loans into lender x approval-FY cohorts.

    Columns: lender, fy, Y, e, n_loans, continuous covariates (means / shares),
    categorical covariates (dominant sector, dominant business type, lender state).
    """
    L = loans[(loans["ApprovalFY"] >= fy_min) & (loans["ApprovalFY"] <= fy_max)].copy()
    L["log_amount"] = np.log(L["GrossApproval"].clip(lower=1))
    g = L.groupby(["LocationID", "ApprovalFY"], sort=True)
    units = g.agg(
        Y=("chgoff", "sum"),
        e=("years_at_risk", "sum"),
        n_loans=("chgoff", "size"),
        chgoff_amount_total=("chgoff_amount", "sum"),
        mean_log_amount=("log_amount", "mean"),
        mean_term_months=("TermInMonths", "mean"),
        guar_share=("guar_share", "mean"),
        mean_jobs=("JobsSupported", "mean"),
        share_revolver=("revolver", "mean"),
        share_express=("express", "mean"),
        share_young=("young_business", "mean"),
        share_collateral=("collateral", "mean"),
        share_franchise=("franchise", "mean"),
        share_sold=("sold_secmkt", "mean"),
    ).reset_index()
    cats = g.agg(sector=("sector", _mode), business_type=("BusinessType", _mode),
                 lender_state=("BankState", _mode), lender_name=("BankName", _mode)).reset_index()
    units = units.merge(cats, on=["LocationID", "ApprovalFY"])
    units = units.rename(columns={"LocationID": "lender", "ApprovalFY": "fy"})
    units["fy"] = units["fy"].astype(int)
    units["lender_state"] = units["lender_state"].fillna("UNK")
    units["Y"] = units["Y"].astype(int)
    return units


def load_data(fy_min: int = 2000, fy_max: int = 2019, use_cache: bool = True) -> pd.DataFrame:
    """Project entry point: one row per lender x approval-FY cohort, columns 'Y' and 'e' + covariates."""
    cache = PROCESSED_DIR / f"units_FY{fy_min}_FY{fy_max}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    units = build_units(load_loans(decades=("2000s", "2010s", "2020s")), fy_min, fy_max)
    units.to_parquet(cache, index=False)
    return units


def summary(units: pd.DataFrame) -> dict:
    """Headline numbers quoted in the data sheet."""
    lender_size = units.groupby("lender")["n_loans"].sum()
    return {
        "units (lender x FY)": len(units),
        "lenders": units["lender"].nunique(),
        "fiscal years": f"{units['fy'].min()}-{units['fy'].max()}",
        "loans": int(units["n_loans"].sum()),
        "charge-offs (sum Y)": int(units["Y"].sum()),
        "loan-years at risk (sum e)": round(float(units["e"].sum())),
        "pooled rate per 1,000 loan-years": round(1000 * units["Y"].sum() / units["e"].sum(), 2),
        "share of units with Y = 0": round(float((units["Y"] == 0).mean()), 3),
        "median Y / median e": f"{units['Y'].median():.0f} / {units['e'].median():.1f}",
        "lenders with < 10 loans": int((lender_size < 10).sum()),
        "largest lender (loans)": int(lender_size.max()),
    }


def load_amounts(fy_min: int = 2000, fy_max: int = 2019) -> pd.DataFrame:
    """Step 11: one row per charged-off loan, gross amount linked to its (lender, fy) unit."""
    L = load_loans(decades=("2000s", "2010s", "2020s"))
    m = (L["chgoff"] == 1) & L["ApprovalFY"].between(fy_min, fy_max)
    out = L.loc[m, ["LocationID", "ApprovalFY", "ChargeOffDate", "chgoff_amount", "GrossApproval"]]
    return out.rename(columns={"LocationID": "lender", "ApprovalFY": "fy"}).reset_index(drop=True)
