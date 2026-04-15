# SurfaceEdge

A hybrid CNN + Embedding model for predicting next-day options price changes
using the full options surface as a visual input alongside contract-specific scalars.

## Repository Structure

| File | Description |
|---|---|
| `build_dataset.py` | Builds surface images and scalar/label files from raw parquets |
| `Dataset.py` | PyTorch Dataset — builds flat contract index, lazy loads files |
| `Model.py` | SurfaceEdge CNN + Embedding model architecture |
| `main.py` | Training and evaluation entry point |
| `download_data.py` | Downloads raw parquet data for all 104 tickers |
| `expand_scalar.py` | Reconstructs full 120-dim scalar array from compressed .npz |
| `baseline_mae.py` | Naive baseline (predict 0.0) MAE evaluation |
| `BSM_pricing.py` | BAW/BSM pricing reference (kept for reference, not used as baseline) |

---

## Step 1 — Download Data

```bash
python download_data.py
```

Populates `data/<ticker>/options.parquet` and `data/<ticker>/underlying.parquet`
for all 104 tickers (~9.4 GB total). Already-downloaded files are skipped automatically.

---

## Step 2 — Build Dataset

```python
from build_dataset import build

build()                      # all 104 tickers
build(ticker='aapl')         # single ticker
build(ticker='aapl', start_date='2020-01-01', end_date='2022-12-31')
```

Or run directly:

```bash
python build_dataset.py
```

Produces the following files per ticker per trading day under `dataset/<ticker>/`:

| File | Description |
|---|---|
| `<date>_calls.png` | Call surface image (60 × 30, RGB) |
| `<date>_puts.png` | Put surface image (60 × 30, RGB) |
| `<date>_calls.npz` | Compressed: `labels` (60×30 float32), `scalars` (60×30×16 float16) |
| `<date>_puts.npz` | Compressed: `labels` (60×30 float32), `scalars` (60×30×16 float16) |

**Surface image channels (RGB):**
- R = implied volatility (normalized to [0, 255])
- G = log1p(open interest) (normalized)
- B = log1p(volume) (normalized)

**Grid dimensions:**
- X axis (WIDTH = 30): days to expiry, 1–61 days
- Y axis (HEIGHT = 60): log-moneyness, ln(strike/spot) from -1.0 to +1.0

**Stock split adjustment:** Strike prices are expressed in post-split terms across
the full dataset history. For each trading date, strikes are multiplied by the
product of all split coefficients occurring after that date. For example, AAPL
contracts before the 4:1 split on August 31, 2020 have strikes multiplied by 4,
and contracts before the 7:1 split on June 9, 2014 are multiplied by 28 (7 × 4).

**Days with fewer than 100 non-zero label cells are skipped** as too sparse.

---

## Step 3 — Train

```bash
python main.py
python main.py --option_type calls --epochs 10 --batch_size 256 --lr 1e-4
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `dataset` | Path to dataset root folder |
| `--split_date` | `2024-11-21` | Chronological train/test cutoff |
| `--option_type` | `calls` | `calls`, `puts`, or `both` |
| `--epochs` | `1` | Number of training epochs |
| `--batch_size` | `256` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--num_workers` | `4` | DataLoader worker count |
| `--save_path` | `surfaceedge.pt` | Checkpoint output path |

The train/test split is **strictly chronological** — all days before `split_date`
are used for training, all days on or after are held out for testing. This prevents
data leakage from the time-series nature of financial data.

---

## Scalar Layout

Each `.npz` stores 16 base scalar features per cell (float16). The OHE ticker
block is excluded from disk and reconstructed at load time from the folder name.

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
| 11 | `dividend_yield` | Annualized dividend yield |
| 12 | `days_to_next_div` | Days until next ex-dividend date, -1 if none |
| 13 | `momentum_5d` | 5-day underlying return, 0.0 if unavailable |
| 14 | `momentum_20d` | 20-day underlying return, 0.0 if unavailable |
| 15 | `implied_vol` | Mean implied volatility for this bin |
| 16–119 | `ticker_ohe` | One-hot ticker (104 elements) — reconstructed at load time |

**Example** (AAPL calls, 2025-01-16, cell (30, 17)):

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

Near-ATM call (strike $230.00 vs spot $227.25), 36 days to expiry, priced at $7.25
with IV of 26.85%. Underlying down 5.95% over 5 days and 9.07% over 20 days.
Next-day mark increased by 5.52%.

**Label:** `(mark_t+1 - mark_t) / mark_t`

---

## Loading Compressed Scalars

To reconstruct the full 120-dim float32 scalar array from a compressed file:

```python
from expand_scalar import load_scalars

scalars = load_scalars('dataset/aapl/2025-01-16_calls.npz')
# scalars.shape -> (60, 30, 120), dtype float32
```

The OHE is reconstructed automatically from the parent folder name.

---

## Baseline

The naive baseline predicts `0.0` (no price change) for every contract.

```python
from baseline_mae import naive_evaluate_dataset

mae, n = naive_evaluate_dataset('dataset/aapl', option_type='calls')
mae, n = naive_evaluate_dataset('dataset/aapl', option_type='puts')
```

| Model | Calls MAE | Puts MAE |
|---|---|---|
| Naive (predict 0.0) | 0.2638 | 0.4642 |
| SurfaceEdge | TBD | TBD |
