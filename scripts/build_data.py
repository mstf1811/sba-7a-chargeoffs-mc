"""Download (if needed), clean and aggregate the SBA 7(a) data, then print the data-sheet numbers.

    python scripts/build_data.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from sba_mc.data import PROCESSED_DIR, load_amounts, load_data, load_loans, summary  # noqa: E402
from sba_mc.download import FILES_DEFAULT, download  # noqa: E402

t0 = time.time()
download(FILES_DEFAULT)
load_loans(use_cache=False)
units = load_data(use_cache=False)
amounts = load_amounts()

print("\nCleaning log (loan level, FY2000-present):")
print(pd.read_csv(PROCESSED_DIR / "cleaning_log_2000s_2010s_2020s.csv").to_string(index=False))
print("\nUnit table (lender x approval FY):")
for k, v in summary(units).items():
    print(f"  {k:<35} {v}")
pos = amounts["chgoff_amount"] > 0
print(f"\nAmounts: {len(amounts):,} charged-off loans, {(~pos).sum()} with amount <= 0, "
      f"median ${amounts.loc[pos, 'chgoff_amount'].median():,.0f}, "
      f"mean ${amounts.loc[pos, 'chgoff_amount'].mean():,.0f}, "
      f"skewness {amounts.loc[pos, 'chgoff_amount'].skew():.2f}")
print(f"\nDone in {time.time() - t0:.0f}s -> {PROCESSED_DIR}")
