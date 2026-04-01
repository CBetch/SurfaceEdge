"""
dataset.py

Builds the processed dataset from raw parquet files.

For each ticker and each trading day, produces:
    dataset/<ticker>/<date>_calls.png
    dataset/<ticker>/<date>_calls_labels.npy
    dataset/<ticker>/<date>_calls_scalars.npy
    dataset/<ticker>/<date>_puts.png
    dataset/<ticker>/<date>_puts_labels.npy
    dataset/<ticker>/<date>_puts_scalars.npy

PNG channels (uint8, 0-255):
    R = implied_volatility          (normalized)
    G = log1p(open_interest)        (normalized)
    B = log1p(volume)               (normalized)

Label matrices (.npy, float32):
    Shape: (HEIGHT, WIDTH)
    Each cell = next-day percentage price change
    label = (mark_t+1 - mark_t) / mark_t
    Cells with no next-day quote are 0.

Scalar matrices (.npy, float32):
    Shape: (HEIGHT, WIDTH, 7)
    Per-cell input features alongside the image:
        [..., 0] = spot          (underlying price)
        [..., 1] = strike        (split-adjusted strike price)
        [..., 2] = tau           (days to expiry)
        [..., 3] = log_moneyness (log(strike/spot))
        [..., 4] = mark          (today's mark price)
        [..., 5] = is_call       (1.0 for call, 0.0 for put)
        [..., 6] = ticker_idx    (integer index into TICKERS list)

Baseline for comparison:
    Predicting 0.0 (no price change) for every cell.

Usage:
    from dataset import build
    build()               # all tickers
    build(ticker='aapl')  # single ticker
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR    = Path(__file__).parent / "data"
DATASET_DIR = Path(__file__).parent / "dataset"

# Surface grid dimensions
WIDTH  = 30   # X axis — expiry bins
HEIGHT = 60   # Y axis — moneyness bins

# Fixed global bin edges — same for every ticker and every day
# log-moneyness: log(strike/spot), 0 = ATM
MONEYNESS_MIN  = -1.0
MONEYNESS_MAX  =  1.0
MONEYNESS_BINS = np.linspace(MONEYNESS_MIN, MONEYNESS_MAX, HEIGHT + 1)

# Days to expiry
TAU_MIN  = 1
TAU_MAX  = 61
TAU_BINS = np.linspace(TAU_MIN, TAU_MAX, WIDTH + 1)

# Minimum number of non-zero label cells required to save a surface
# Days below this threshold are skipped as too sparse to be useful
MIN_NONZERO_CELLS = 100

TICKERS = [
    "aapl", "abbv", "abt",   "acn",   "adbe", "aig",  "amd",  "amgn", "amt",  "amzn",
    "avgo", "axp",  "ba",    "bac",   "bk",   "bkng", "blk",  "bmy",  "brk.b","c",
    "cat",  "cl",   "cmcsa", "cof",   "cop",  "cost", "crm",  "csco", "cvs",  "cvx",
    "de",   "dhr",  "dis",   "duk",   "emr",  "fdx",  "gd",   "ge",   "gild", "gm",
    "goog", "googl","gs",    "hd",    "hon",  "ibm",  "intu", "isrg", "iwm",  "jnj",
    "jpm",  "ko",   "lin",   "lly",   "lmt",  "low",  "ma",   "mcd",  "mdlz", "mdt",
    "met",  "meta", "mmm",   "mo",    "mrk",  "ms",   "msft", "nee",  "nflx", "nke",
    "now",  "nvda", "orcl",  "pep",   "pfe",  "pg",   "pltr", "pm",   "pypl", "qcom",
    "qqq",  "rtx",  "sbux",  "schw",  "so",   "spg",  "spy",  "t",    "tgt",  "tmo",
    "tmus", "tsla", "txn",   "uber",  "unh",  "unp",  "ups",  "usb",  "v",    "vix",
    "vz",   "wfc",  "wmt",   "xom",
]

# Integer index lookup for ticker encoding in scalars
TICKER_IDX = {t: i for i, t in enumerate(TICKERS)}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1], handling constant arrays."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _build_surface(
    snapshot: pd.DataFrame,
    spot: float,
    is_call: bool,
    ticker_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Convert a single-day, single-type options snapshot into fixed-size grids.

    snapshot must contain:
        - 'label' column: percentage price change (mark_t+1 - mark_t) / mark_t
        - 'mark'  column: today's mark price
        - 'strike' column: split-adjusted strike price

    Returns:
        image:         (HEIGHT, WIDTH, 3) uint8
        label_grid:    (HEIGHT, WIDTH)    float32 — next-day % price change
        scalar_grid:   (HEIGHT, WIDTH, 7) float32 — [spot, strike, tau,
                                                      log_money, mark,
                                                      is_call, ticker_idx]

    Returns None if the snapshot is empty or has insufficient data.
    """
    df = snapshot.copy()
    df["date"]       = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["tau"]        = (df["expiration"] - df["date"]).dt.days

    # Filter to valid expiry window
    df = df[(df["tau"] >= TAU_MIN) & (df["tau"] <= TAU_MAX)]
    if df.empty:
        return None

    # Compute log-moneyness: log(strike/spot), centered at 0 = ATM
    df["log_money"] = np.log(df["strike"] / spot)

    # Assign each row to a fixed global bin (options outside range are dropped)
    df["x_bin"] = pd.cut(df["tau"],       bins=TAU_BINS,       labels=False, include_lowest=True)
    df["y_bin"] = pd.cut(df["log_money"], bins=MONEYNESS_BINS, labels=False, include_lowest=True)
    df = df.dropna(subset=["x_bin", "y_bin"])
    df["x_bin"] = df["x_bin"].astype(int)
    df["y_bin"] = df["y_bin"].astype(int)

    if df.empty:
        return None

    # Aggregate per pixel
    agg = df.groupby(["y_bin", "x_bin"], observed=True).agg(
        iv        = ("implied_volatility", "mean"),
        oi        = ("open_interest",      "sum"),
        vol       = ("volume",             "sum"),
        label     = ("label",              "mean"),
        mark      = ("mark",               "mean"),
        strike    = ("strike",             "mean"),
        tau       = ("tau",                "mean"),
        log_money = ("log_money",          "mean"),
    ).reset_index()

    # Build grids
    iv_grid     = np.zeros((HEIGHT, WIDTH),    dtype=np.float32)
    oi_grid     = np.zeros((HEIGHT, WIDTH),    dtype=np.float32)
    vol_grid    = np.zeros((HEIGHT, WIDTH),    dtype=np.float32)
    label_grid  = np.zeros((HEIGHT, WIDTH),    dtype=np.float32)
    scalar_grid = np.zeros((HEIGHT, WIDTH, 7), dtype=np.float32)

    rows = agg["y_bin"].values
    cols = agg["x_bin"].values

    iv_grid[rows, cols]        = agg["iv"].values
    oi_grid[rows, cols]        = np.log1p(agg["oi"].values)
    vol_grid[rows, cols]       = np.log1p(agg["vol"].values)
    label_grid[rows, cols]     = agg["label"].values
    scalar_grid[rows, cols, 0] = float(spot)
    scalar_grid[rows, cols, 1] = agg["strike"].values
    scalar_grid[rows, cols, 2] = agg["tau"].values
    scalar_grid[rows, cols, 3] = agg["log_money"].values
    scalar_grid[rows, cols, 4] = agg["mark"].values
    scalar_grid[rows, cols, 5] = float(is_call)
    scalar_grid[rows, cols, 6] = float(ticker_idx)

    # Normalize image channels and pack into uint8
    r = (_normalize(iv_grid)  * 255).astype(np.uint8)
    g = (_normalize(oi_grid)  * 255).astype(np.uint8)
    b = (_normalize(vol_grid) * 255).astype(np.uint8)
    image = np.stack([r, g, b], axis=-1)  # (HEIGHT, WIDTH, 3)

    return image, label_grid, scalar_grid


def _process_ticker(ticker: str, start_date: str | None = None, end_date: str | None = None) -> None:
    """Load raw parquets for one ticker and write PNGs, labels, and scalars."""
    options_path    = DATA_DIR / ticker / "options.parquet"
    underlying_path = DATA_DIR / ticker / "underlying.parquet"

    if not options_path.exists() or not underlying_path.exists():
        print(f"  [{ticker}] missing parquet files, skipping")
        return

    print(f"  [{ticker}] loading parquets ...")
    opts = pd.read_parquet(options_path)
    und  = pd.read_parquet(underlying_path)

    if opts.empty:
        print(f"  [{ticker}] options file is empty, skipping")
        return

    opts["date"]       = pd.to_datetime(opts["date"])
    opts["expiration"] = pd.to_datetime(opts["expiration"])
    und["date"]        = pd.to_datetime(und["date"])

    # Build cumulative split adjustment: for each date, multiply by all
    # split coefficients that occurred AFTER that date to convert strikes
    # to post-split terms, making all historical data comparable
    splits = und[und["split_coefficient"] != 1.0][["date", "split_coefficient"]].copy()
    splits["split_coefficient"] = splits["split_coefficient"].round()
    splits = splits.sort_values("date")

    unique_dates = sorted(opts["date"].unique())
    adjustment_map = {}
    for d in unique_dates:
        future = splits[splits["date"] > d]["split_coefficient"]
        adjustment_map[d] = float(future.prod()) if not future.empty else 1.0

    opts["strike"] = opts["strike"] * opts["date"].map(adjustment_map)

    # Compute next-day percentage price change label
    print(f"  [{ticker}] computing next-day labels ...")
    mark_lookup = opts[["contract_id", "date", "mark"]].copy()
    mark_lookup = mark_lookup.rename(columns={"mark": "next_mark", "date": "next_date"})

    trading_dates = sorted(opts["date"].unique())
    next_date_map = {d: trading_dates[i + 1] for i, d in enumerate(trading_dates[:-1])}
    opts["next_date"] = opts["date"].map(next_date_map)

    opts = opts.merge(
        mark_lookup,
        left_on=["contract_id", "next_date"],
        right_on=["contract_id", "next_date"],
        how="left",
    )

    # Percentage change: (mark_t+1 - mark_t) / mark_t
    # Drop rows where mark is zero or next_mark is unavailable
    opts = opts[opts["next_mark"].notna() & (opts["mark"] > 0)].copy()
    opts["label"] = (opts["next_mark"] - opts["mark"]) / opts["mark"]

    if opts.empty:
        print(f"  [{ticker}] no labellable rows, skipping")
        return

    out_dir    = DATASET_DIR / ticker
    ticker_idx = TICKER_IDX[ticker]
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = sorted(opts["date"].unique())
    if start_date:
        dates = [d for d in dates if pd.Timestamp(d).strftime("%Y-%m-%d") >= start_date]
    if end_date:
        dates = [d for d in dates if pd.Timestamp(d).strftime("%Y-%m-%d") <= end_date]
    print(f"  [{ticker}] building surfaces for {len(dates)} trading days ...")

    for date in tqdm(dates, desc=f"  [{ticker}]", unit="day"):
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        day_opts = opts[opts["date"] == date]
        spot_row = und[und["date"] == date]

        if spot_row.empty:
            continue
        spot = float(spot_row["adjusted_close"].iloc[0])

        for option_type in ("calls", "puts"):
            png_path    = out_dir / f"{date_str}_{option_type}.png"
            label_path  = out_dir / f"{date_str}_{option_type}_labels.npy"
            scalar_path = out_dir / f"{date_str}_{option_type}_scalars.npy"

            if png_path.exists() and label_path.exists() and scalar_path.exists():
                continue

            is_call  = option_type == "calls"
            type_str = "call" if is_call else "put"
            snapshot = day_opts[day_opts["type"] == type_str]

            result = _build_surface(snapshot, spot, is_call, ticker_idx)
            if result is None:
                continue

            image, label_grid, scalar_grid = result

            # Skip if too few contracts have next-day labels
            if np.count_nonzero(label_grid) < MIN_NONZERO_CELLS:
                continue

            Image.fromarray(image).save(png_path)
            np.save(label_path,  label_grid)
            np.save(scalar_path, scalar_grid)

    print(f"  [{ticker}] done")


# ── Public API ────────────────────────────────────────────────────────────────

def build(
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """
    Build the processed dataset from raw parquet files.

    Args:
        ticker:     Lowercase ticker string (e.g. 'aapl').
                    If None, processes all tickers.
        start_date: Only process days on or after this date ('YYYY-MM-DD').
        end_date:   Only process days on or before this date ('YYYY-MM-DD').
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if ticker is not None:
        ticker = ticker.lower()
        if ticker not in TICKERS:
            raise ValueError(f"Unknown ticker '{ticker}'. Must be one of: {TICKERS}")
        targets = [ticker]
    else:
        targets = TICKERS

    print(f"Building dataset for {len(targets)} ticker(s) ...")
    for t in targets:
        _process_ticker(t, start_date=start_date, end_date=end_date)
    print("\nDataset build complete.")


if __name__ == "__main__":
    import shutil

    if DATASET_DIR.exists():
        answer = input(f"Delete existing dataset directory '{DATASET_DIR}'? [y/N]: ").strip().lower()
        if answer == "y":
            shutil.rmtree(DATASET_DIR)
            print(f"Deleted {DATASET_DIR}")
        else:
            print("Keeping existing files — already-complete days will be skipped.")

    build()