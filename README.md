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
| `<date>_calls_labels.npy` | Next-day mark price per grid cell (60 × 30) |
| `<date>_puts_labels.npy` | Next-day mark price per grid cell (60 × 30) |
| `<date>_calls_scalars.npy` | Per-cell scalar features (60 × 30 × 4) |
| `<date>_puts_scalars.npy` | Per-cell scalar features (60 × 30 × 4) |
| `<date>_calls_meta.json` | Ticker, date, spot price |
| `<date>_puts_meta.json` | Ticker, date, spot price |

**Surface image channels (RGB):**
- R = implied volatility (normalized)
- G = log1p(open interest) (normalized)
- B = log1p(volume) (normalized)

**Grid dimensions:**
- X axis (WIDTH = 30): expiry bins, 1–61 days to expiry
- Y axis (HEIGHT = 60): log-moneyness bins, log(strike/spot) from -0.5 to +1.5

## Model

Each training sample is a single option contract on a single day. The model takes:

- **Surface image** `(3 × 60 × 30)` — the full options surface for that ticker/day, providing global market context
- **Scalar features** `(4,)` — contract-specific inputs:
  - `tau` — days to expiry
  - `log_moneyness` — log(strike / spot)
  - `is_call` — 1.0 for call, 0.0 for put
  - `mark` — today's bid/ask midpoint price

**Label:** next-day mark price for that contract

**Baseline:** predicting `mark_t+1 = mark_t` (prices don't change). The model must achieve a lower MAE than this naive baseline to demonstrate the surface image contains useful predictive information.