# Inference and Monte Carlo on SBA 7(a) charge-offs

Semester project: count models, inference and Monte Carlo on real data.
Unit = lender × approval fiscal-year cohort; Y = charge-offs; e = loan-years at risk.
See [docs/data_sheet.md](docs/data_sheet.md) for the source, licence, role mapping and quality issues.

## Rerun everything

```bash
pip install -r requirements.txt
python scripts/build_data.py      # downloads ~760 MB to data/raw/, builds data/processed/*.parquet
pytest                            # tests for the data pipeline and simulation module
```

Raw data is not committed. `data/raw/manifest.json` records the URL, size and SHA-256 of each downloaded file.

## Layout

```
src/sba_mc/download.py   download the FOIA files from data.sba.gov
src/sba_mc/data.py       clean loans -> lender x FY units (load_data), amounts (load_amounts)
src/sba_mc/simulate.py   vectorised Monte Carlo functions (tested)
scripts/build_data.py    end-to-end data build + data-sheet numbers
notebooks/               one notebook per step (01 = step 1: checks, descriptives, model, sufficiency)
docs/                    data sheet, model note (step 1), figures, reports
tests/                   pytest
```

## Python usage

```python
from sba_mc.data import load_data, load_amounts
df = load_data()          # 32,806 rows, columns Y, e, covariates
amounts = load_amounts()  # 178,803 charged-off loans with gross amount
```

Every Monte Carlo experiment uses one `numpy.random.default_rng(seed)` with its seed reported.
