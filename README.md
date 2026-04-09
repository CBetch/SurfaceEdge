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
| `<date>_calls.npz` | Compressed archive: `labels` (60 × 30 float32), `scalars` (60 × 30 × 16 float16) |
| `<date>_puts.npz` | Compressed archive: `labels` (60 × 30 float32), `scalars` (60 × 30 × 16 float16) |

The OHE ticker vector is **not stored on disk** — it is reconstructed at load time from the filename using `TICKER_IDX` from `dataset.py`. Use `load_surface()` from `compress.py` to load files with the full 120-dim scalar vector automatically reconstructed.

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
- **Scalar vector** `(120,)` — contract-specific inputs:

| Index | Feature | Description |
|---|---|---|
| 0 | `spot` | Underlying price at snapshot time |
| 1 | `strike` | Split-adjusted strike price |
| 2 | `tau` | Days to expiry |
| 3 | `log_moneyness` | ln(strike / spot) |
| 4 | `mark` | Today's bid/ask midpoint price |
| 5 | `is_call` | 1.0 for call, 0.0 for put |
| 6 | `delta` | Option delta |
| 7 | `gamma` | Option gamma |
| 8 | `vega` | Option vega |
| 9 | `theta` | Option theta |
| 10 | `spread` | Normalized bid-ask spread: (ask - bid) / mark |
| 11 | `dividend_yield` | Annualized dividend yield of the underlying |
| 12 | `days_to_next_div` | Days until next ex-dividend date, -1 if none |
| 13 | `momentum_5d` | 5-day underlying return, 0.0 if unavailable |
| 14 | `momentum_20d` | 20-day underlying return, 0.0 if unavailable |
| 15 | `implied_vol` | Mean implied volatility for this bin |
| 16–119 | `ticker_ohe` | One-hot encoded ticker (104 elements) |

**Example scalar vector** (AAPL calls, 2025-01-16, cell (30, 17)):

```
  [0]  spot             : 227.2491
  [1]  strike           : 230.0000
  [2]  tau              : 36.0 days
  [3]  log_moneyness    : 0.0120
  [4]  mark             : 7.2500
  [5]  is_call          : 1
  [6]  delta            : 0.5011
  [7]  gamma            : 0.0207
  [8]  vega             : 0.2860
  [9]  theta            : -0.1194
  [10] spread           : 0.0276
  [11] dividend_yield   : 0.0044
  [12] days_to_div      : 25
  [13] momentum_5d      : -0.0595
  [14] momentum_20d     : -0.0907
  [15] implied_vol      : 0.2685
  [16:120] OHE sum      : 1  (should be 1)
  [16:120] OHE hot idx  : 0  (aapl)
  label                 : 0.055172  (5.5172%)
```

This represents a near-ATM call (strike $230.00 vs spot $227.25) expiring in 36 days, priced at $7.25 with an IV of 26.85%. The underlying has been declining over both the 5-day (-5.95%) and 20-day (-9.07%) windows. The next day this contract's mark price increased by 5.52%.

**Label:** next-day percentage price change — `(mark_t+1 - mark_t) / mark_t`

**Baseline:** predicting `0.0` (no price change) for every contract. The model must achieve a lower MAE than this naive baseline to demonstrate the surface image contains useful predictive information.

## Loading Compressed Scalars

Scalar files are stored as compressed `.npz` with float16 precision and without the OHE ticker block to reduce disk size. Use `expand_scalar.py` to reconstruct the full 120-dim float32 scalar array identical to the pre-compression format:

```python
from expand_scalar import load_scalars

scalars = load_scalars('dataset/aapl/2025-01-16_calls.npz')
# scalars.shape -> (60, 30, 120), dtype float32
```

The ticker OHE is reconstructed automatically from the folder name — no extra arguments needed.

## Baseline

The naive baseline predicts `0.0` (no price change) for every contract. MAE under this baseline equals the mean absolute value of all labels — the minimum bar SurfaceEdge must beat to demonstrate that surface images contain useful predictive information.

```python
from baseline import naive_predict_file, naive_evaluate_dataset

# Single file
mae, n = naive_predict_file(
    'dataset/aapl/2025-01-16_calls_scalars.npy',
    'dataset/aapl/2025-01-16_calls_labels.npy',
    verbose=True,
)

# Full ticker dataset — calls, puts, or both
mae, n = naive_evaluate_dataset('dataset/aapl')
mae, n = naive_evaluate_dataset('dataset/aapl', option_type='calls')
mae, n = naive_evaluate_dataset('dataset/aapl', option_type='puts')