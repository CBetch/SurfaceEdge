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
    Shape: (HEIGHT, WIDTH, 15 + len(TICKERS))  =  (HEIGHT, WIDTH, 119)
    Per-cell input features alongside the image:
        [..., 0]      = spot             (underlying price)
        [..., 1]      = strike           (split-adjusted strike price)
        [..., 2]      = tau              (days to expiry)
        [..., 3]      = log_moneyness    (ln(strike/spot))
        [..., 4]      = mark             (today's mark price)
        [..., 5]      = is_call          (1.0 for call, 0.0 for put)
        [..., 6]      = delta
        [..., 7]      = gamma
        [..., 8]      = vega
        [..., 9]      = theta
        [..., 10]     = spread           (normalized bid-ask spread: (ask-bid)/mark)
        [..., 11]     = dividend_yield   (annualized dividend yield)
        [..., 12]     = days_to_div      (days to next dividend, -1 if none)
        [..., 13]     = momentum_5d      (5-day underlying return, 0 if unavailable)
        [..., 14]     = momentum_20d     (20-day underlying return, 0 if unavailable)
        [..., 15]     = implied_volatility (mean IV for this bin)
        [..., 16:120] = one-hot ticker encoding (104 elements, one per ticker)

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
MIN_NONZERO_CELLS = 100

# Number of non-OHE scalars
N_BASE_SCALARS = 16

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

TICKER_IDX = {t: i for i, t in enumerate(TICKERS)}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1], handling constant arrays."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _build_underlying_features(und: pd.DataFrame, date: pd.Timestamp) -> dict:
    """
    Compute underlying-level features for a given date from the underlying DataFrame.
    Returns a dict with: spot, dividend_yield, days_to_div, momentum_5d, momentum_20d.
    """
    past = und[und["date"] <= date].sort_values("date")
    if past.empty:
        return {"spot": 0.0, "dividend_yield": 0.0, "days_to_div": -1,
                "momentum_5d": 0.0, "momentum_20d": 0.0}

    spot = float(past.iloc[-1]["adjusted_close"])

    # Dividend yield — annualize using most recent non-zero dividend
    divs = past[past["dividend_amount"] > 0]
    if not divs.empty:
        last_div = float(divs.iloc[-1]["dividend_amount"])
        # Estimate payments per year from spacing of dividends
        if len(divs) >= 2:
            gaps = divs["date"].diff().dropna().dt.days
            avg_gap = float(gaps.mean())
            payments_per_year = 365.0 / avg_gap if avg_gap > 0 else 4.0
        else:
            payments_per_year = 4.0  # assume quarterly
        div_yield = (last_div * payments_per_year) / spot if spot > 0 else 0.0
    else:
        div_yield = 0.0

    # Days to next dividend
    future_divs = und[(und["date"] > date) & (und["dividend_amount"] > 0)].sort_values("date")
    if not future_divs.empty:
        days_to_div = (future_divs.iloc[0]["date"] - date).days
    else:
        days_to_div = -1

    # Momentum — 5 and 20 day returns
    def _momentum(n):
        if len(past) > n:
            prev = float(past.iloc[-(n+1)]["adjusted_close"])
            return (spot - prev) / prev if prev > 0 else 0.0
        return 0.0

    return {
        "spot":           spot,
        "dividend_yield": float(div_yield),
        "days_to_div":    int(days_to_div),
        "momentum_5d":    _momentum(5),
        "momentum_20d":   _momentum(20),
    }


def _build_surface(
    snapshot: pd.DataFrame,
    spot: float,
    is_call: bool,
    ticker_idx: int,
    div_yield: float,
    days_to_div: int,
    momentum_5d: float,
    momentum_20d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Convert a single-day, single-type options snapshot into fixed-size grids.

    Returns:
        image:         (HEIGHT, WIDTH, 3)                uint8
        label_grid:    (HEIGHT, WIDTH)                   float32
        scalar_grid:   (HEIGHT, WIDTH, N_BASE_SCALARS + len(TICKERS)) float32
    """
    df = snapshot.copy()
    df["date"]       = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["tau"]        = (df["expiration"] - df["date"]).dt.days

    df = df[(df["tau"] >= TAU_MIN) & (df["tau"] <= TAU_MAX)]
    if df.empty:
        return None

    df = df[df["strike"] > 0]
    if df.empty:
        return None

    df["log_money"] = np.log(df["strike"] / spot)
    df["spread"]    = np.where(
        df["mark"] > 0,
        (df["ask"] - df["bid"]) / df["mark"],
        0.0
    )

    df["x_bin"] = pd.cut(df["tau"],       bins=TAU_BINS,       labels=False, include_lowest=True)
    df["y_bin"] = pd.cut(df["log_money"], bins=MONEYNESS_BINS, labels=False, include_lowest=True)
    df = df.dropna(subset=["x_bin", "y_bin"])
    df["x_bin"] = df["x_bin"].astype(int)
    df["y_bin"] = df["y_bin"].astype(int)

    if df.empty:
        return None

    agg = df.groupby(["y_bin", "x_bin"], observed=True).agg(
        iv        = ("implied_volatility", "mean"),
        oi        = ("open_interest",      "sum"),
        vol       = ("volume",             "sum"),
        label     = ("label",              "mean"),
        mark      = ("mark",               "mean"),
        strike    = ("strike",             "mean"),
        tau       = ("tau",                "mean"),
        log_money = ("log_money",          "mean"),
        delta     = ("delta",              "mean"),
        gamma     = ("gamma",              "mean"),
        vega      = ("vega",               "mean"),
        theta     = ("theta",              "mean"),
        spread    = ("spread",             "mean"),
    ).reset_index()

    iv_grid     = np.zeros((HEIGHT, WIDTH),                        dtype=np.float32)
    oi_grid     = np.zeros((HEIGHT, WIDTH),                        dtype=np.float32)
    vol_grid    = np.zeros((HEIGHT, WIDTH),                        dtype=np.float32)
    label_grid  = np.zeros((HEIGHT, WIDTH),                        dtype=np.float32)
    n_scalars   = N_BASE_SCALARS + len(TICKERS)
    scalar_grid = np.zeros((HEIGHT, WIDTH, n_scalars),             dtype=np.float32)

    rows = agg["y_bin"].values
    cols = agg["x_bin"].values

    iv_grid[rows, cols]         = agg["iv"].values
    oi_grid[rows, cols]         = np.log1p(agg["oi"].values)
    vol_grid[rows, cols]        = np.log1p(agg["vol"].values)
    label_grid[rows, cols]      = agg["label"].values

    scalar_grid[rows, cols,  0] = float(spot)
    scalar_grid[rows, cols,  1] = agg["strike"].values
    scalar_grid[rows, cols,  2] = agg["tau"].values
    scalar_grid[rows, cols,  3] = agg["log_money"].values
    scalar_grid[rows, cols,  4] = agg["mark"].values
    scalar_grid[rows, cols,  5] = float(is_call)
    scalar_grid[rows, cols,  6] = np.nan_to_num(agg["delta"].values,  nan=0.0)
    scalar_grid[rows, cols,  7] = np.nan_to_num(agg["gamma"].values,  nan=0.0)
    scalar_grid[rows, cols,  8] = np.nan_to_num(agg["vega"].values,   nan=0.0)
    scalar_grid[rows, cols,  9] = np.nan_to_num(agg["theta"].values,  nan=0.0)
    scalar_grid[rows, cols, 10] = np.nan_to_num(agg["spread"].values, nan=0.0)
    scalar_grid[rows, cols, 11] = float(div_yield)
    scalar_grid[rows, cols, 12] = float(days_to_div)
    scalar_grid[rows, cols, 13] = float(momentum_5d)
    scalar_grid[rows, cols, 14] = float(momentum_20d)
    scalar_grid[rows, cols, 15] = np.nan_to_num(agg["iv"].values, nan=0.0)
    scalar_grid[rows, cols, N_BASE_SCALARS + ticker_idx] = 1.0

    iv_grid  = np.nan_to_num(iv_grid,  nan=0.0)
    oi_grid  = np.nan_to_num(oi_grid,  nan=0.0)
    vol_grid = np.nan_to_num(vol_grid, nan=0.0)

    r = (_normalize(iv_grid)  * 255).astype(np.uint8)
    g = (_normalize(oi_grid)  * 255).astype(np.uint8)
    b = (_normalize(vol_grid) * 255).astype(np.uint8)
    image = np.stack([r, g, b], axis=-1)

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
    und                = und.sort_values("date").reset_index(drop=True)

    # Split adjustment
    splits = und[und["split_coefficient"] != 1.0][["date", "split_coefficient"]].copy()
    splits["split_coefficient"] = splits["split_coefficient"].round()
    splits = splits.sort_values("date")

    unique_dates = sorted(opts["date"].unique())
    adjustment_map = {}
    for d in unique_dates:
        future = splits[splits["date"] > d]["split_coefficient"]
        adjustment_map[d] = float(future.prod()) if not future.empty else 1.0

    opts["strike"] = opts["strike"] * opts["date"].map(adjustment_map)

    # Next-day label
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

        # Compute underlying features once per day
        uf = _build_underlying_features(und, pd.Timestamp(date))
        spot = uf["spot"]
        if spot <= 0:
            continue

        for option_type in ("calls", "puts"):
            png_path    = out_dir / f"{date_str}_{option_type}.png"
            label_path  = out_dir / f"{date_str}_{option_type}_labels.npy"
            scalar_path = out_dir / f"{date_str}_{option_type}_scalars.npy"

            if png_path.exists() and label_path.exists() and scalar_path.exists():
                continue

            is_call  = option_type == "calls"
            type_str = "call" if is_call else "put"
            snapshot = day_opts[day_opts["type"] == type_str]

            result = _build_surface(
                snapshot, spot, is_call, ticker_idx,
                uf["dividend_yield"], uf["days_to_div"],
                uf["momentum_5d"],    uf["momentum_20d"],
            )
            if result is None:
                continue

            image, label_grid, scalar_grid = result

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

    build(ticker="msft")
    build(ticker="aapl")
    build(ticker="googl")
    build(ticker="amzn")
