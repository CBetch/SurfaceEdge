# SurfaceEdge

## Data

This project uses historical options chain data for 104 US equities and ETFs (2008–2025), sourced from [philippdubach/options-data](https://github.com/philippdubach/options-data).

Data is not included in the repo. Run the download script once before anything else:

```bash
python download_data.py
```

This will populate `data/<ticker>/options.parquet` and `data/<ticker>/underlying.parquet` for all 104 tickers. Already-downloaded files are skipped automatically, so re-running is safe.

## Building the Dataset

Once data is downloaded, run the dataset builder to generate surface images and labels:

```python
from dataset import build

build()                      # all 104 tickers
build(ticker='aapl')         # single ticker
build(ticker='aapl', start_date='2020-01-01', end_date='2022-12-31')  # date range
```

Or run directly:

```bash
python dataset.py
```

This produces the following files per ticker per trading day under `dataset/<ticker>/`:

| File | Description |
|---|---|
| `<date>_calls.png` | Options surface image for calls (60 × 30 × 3) |
| `<date>_puts.png` | Options surface image for puts (60 × 30 × 3) |
| `<date>_calls_labels.npy` | Next-day percentage price change per grid cell (60 × 30) |
| `<date>_puts_labels.npy` | Next-day percentage price change per grid cell (60 × 30) |
| `<date>_calls_scalars.npy` | Per-cell scalar features (60 × 30 × 110) |
| `<date>_puts_scalars.npy` | Per-cell scalar features (60 × 30 × 110) |

**Surface image channels (RGB):**
- R = implied volatility (normalized)
- G = log1p(open interest) (normalized)
- B = log1p(volume) (normalized)

**Grid dimensions:**
- X axis (WIDTH = 30): expiry bins, 1–61 days to expiry
- Y axis (HEIGHT = 60): log-moneyness bins, ln(strike/spot) from -1.0 to +1.0

**Stock split adjustment:** Strike prices are adjusted for historical stock splits so that all data is expressed in post-split terms. For each trading date, the strike is multiplied by the product of all split coefficients that occurred after that date. For example, AAPL contracts from before the 4:1 split on August 31, 2020 have their strikes multiplied by 4, and contracts from before the 7:1 split on June 9, 2014 are multiplied by 28 (7 × 4). This ensures log-moneyness is consistent across the full dataset history.

**Days with fewer than 100 non-zero label cells are skipped** as too sparse to provide useful training signal.

## Model

Each training sample is a single option contract on a single day. The model takes:

- **Surface image** `(3 × 60 × 30)` — the full options surface for that ticker/day, providing global market context
- **Scalar features** `(110,)` — contract-specific inputs:
  - `spot` — underlying price at snapshot time
  - `strike` — split-adjusted strike price
  - `tau` — days to expiry
  - `log_moneyness` — ln(strike / spot)
  - `mark` — today's bid/ask midpoint price
  - `is_call` — 1.0 for call, 0.0 for put
  - `ticker` — one-hot encoded vector of length 104, one element per ticker

**Label:** next-day percentage price change — `(mark_t+1 - mark_t) / mark_t`

**Baseline:** predicting `0.0` (no price change) for every contract. The model must achieve a lower MAE than this naive baseline to demonstrate the surface image contains useful predictive information.