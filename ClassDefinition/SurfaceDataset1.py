"""
Dataset.py

PyTorch Dataset for SurfaceEdge. Builds a flat contract-level index at startup
then lazily loads files on demand during training.

Each sample corresponds to a single option contract on a single trading day.
The model receives:
    image         : (3, 60, 30) float32  — full options surface for that day
    tau           : scalar float32        — days to expiry
    log_moneyness : scalar float32        — ln(strike/spot)
    is_call       : scalar float32        — 1.0 for call, 0.0 for put
    mark          : scalar float32        — today's bid/ask midpoint
    ticker_idx    : scalar int64          — integer ticker index for embedding
    label         : scalar float32        — next-day % price change

Train/test split is strictly chronological to prevent data leakage.
The split date is fixed globally across all tickers.

Usage:
    from Dataset import SurfaceDataset
    from torch.utils.data import DataLoader

    train_ds = SurfaceDataset('dataset', split='train')
    test_ds  = SurfaceDataset('dataset', split='test')

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

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

# Chronological train/test split date — all days before this are train,
# on or after are test. Based on 80/20 split of the full AAPL dataset.
SPLIT_DATE = "2024-11-21"

# Minimum non-zero label cells required — mirrors build_dataset.py filter
MIN_NONZERO_CELLS = 100

# ── Normalization Constants ───────────────────────────────────────────────────
IMG_MEAN = torch.tensor([0.025323, 0.052204, 0.029041]).view(3, 1, 1)
IMG_STD  = torch.tensor([0.092316, 0.178988, 0.123127]).view(3, 1, 1)

TAU_MEAN,  TAU_STD  = 26.707440, 15.572564
LM_MEAN,   LM_STD   = -0.079088,  0.324890
MARK_MEAN, MARK_STD = 38.741629, 42.979143

STATS_MEAN = torch.tensor([
    0.164838, 0.009879, 0.065479, -0.063699, 0.079269,
    2.697221, 80.434174, 0.004126, 0.016407, 0.518540
])
STATS_STD  = torch.tensor([
    0.774911, 0.016821, 0.081800, 0.099738, 0.094645,
    6.855298, 137.171036, 0.041085, 0.076115, 0.520742
])

# ── Dataset ───────────────────────────────────────────────────────────────────

class SurfaceDataset(Dataset):
    """
    Flat contract-level dataset for SurfaceEdge.

    Scans all ticker subdirectories for .npz files, applies the chronological
    split, then builds a flat index of (npz_path, y, x, ticker_idx) tuples —
    one entry per valid contract. Files are loaded lazily at __getitem__ time.

    Args:
        dataset_dir  : path to the dataset/ root folder
        split        : 'train' or 'test'
        split_date   : cutoff date string 'YYYY-MM-DD' (train < split_date <= test)
        option_type  : 'calls', 'puts', or 'both'
    """

    def __init__(
        self,
        dataset_dir: str,
        split:       str = "train",
        split_date:  str = SPLIT_DATE,
        option_type: str = "both",
        ticker_list: list = TICKERS
    ):
        super().__init__()
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        assert option_type in ("calls", "puts", "both"), "option_type must be 'calls', 'puts', or 'both'"

        self.dataset_dir = Path(dataset_dir)
        self.split       = split
        self.split_date  = split_date
        self.option_type = option_type

        # Determine which surface types to include
        types = []
        if option_type in ("calls", "both"):
            types.append("calls")
        if option_type in ("puts", "both"):
            types.append("puts")

        # Build flat index: list of (npz_path, y, x, ticker_idx)
        print(f"Building {split} index (split_date={split_date}, option_type={option_type}) ...")
        self.samples = []

        self.local_ticker_idx = {t: i for i, t in enumerate(ticker_list)}


        for ticker_dir in sorted(self.dataset_dir.iterdir()):
            if not ticker_dir.is_dir():
                continue

            ticker = ticker_dir.name.lower()
            if ticker not in TICKER_IDX or ticker not in ticker_list:
                continue
            # ticker_idx = TICKER_IDX[ticker]
            ticker_idx = self.local_ticker_idx[ticker]

            for opt_type in types:
                for npz_path in sorted(ticker_dir.glob(f"*_{opt_type}.npz")):
                    # Extract date from filename e.g. '2020-09-01_calls.npz'
                    date_str = npz_path.stem.replace(f"_{opt_type}", "")

                    # Chronological split
                    if split == "train" and date_str >= split_date:
                        continue
                    if split == "test" and date_str < split_date:
                        continue

                    # Verify image exists
                    png_path = npz_path.with_suffix(".png")
                    if not png_path.exists():
                        continue

                    # Load labels to find valid contracts
                    data   = np.load(npz_path)
                    labels = data["labels"]
                    ys, xs = np.where(labels != 0)

                    if len(ys) < MIN_NONZERO_CELLS:
                        continue

                    for y, x in zip(ys, xs):
                        self.samples.append((str(npz_path), int(y), int(x), ticker_idx))

        print(f"  {split}: {len(self.samples):,} contracts across all tickers")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        npz_path_str, y, x, ticker_idx = self.samples[idx]
        npz_path = Path(npz_path_str)

        # ── Load surface image ────────────────────────────────────────────────
        png_path = npz_path.with_suffix(".png")
        image    = read_image(str(png_path)).float() / 255.0  # (3, 60, 30) float32

        # ── Load scalars and label ────────────────────────────────────────────
        # data    = np.load(npz_path)
        # scalars = data["scalars"].astype(np.float32)  # (60, 30, 16) float16 -> float32
        # labels  = data["labels"]                       # (60, 30) float32

        # cell_scalars = scalars[y, x]   # (16,)
        # cell_label   = float(labels[y, x])
        # ── Load scalars and label via Memory Mapping (SLOW load fix) ─────────────────────────
        npy_scalars_path = npz_path.with_name(f"{npz_path.stem}_scalars.npy")
        npy_labels_path  = npz_path.with_name(f"{npz_path.stem}_labels.npy")

        # The magic keyword: mmap_mode='r'. This treats the file on disk like an array
        # in RAM without actually loading the whole thing.
        scalars_mmap = np.load(npy_scalars_path, mmap_mode='r')
        labels_mmap  = np.load(npy_labels_path, mmap_mode='r')

        # Extract only the exact [y, x] coordinate we need.
        # We wrap in np.array() to pull it from disk into a real numpy array.
        cell_scalars = np.array(scalars_mmap[y, x])
        cell_label   = float(labels_mmap[y, x])

        # ── Extract named scalars ─────────────────────────────────────────────
        # Scalar layout: [0]=spot [1]=strike [2]=tau [3]=log_moneyness
        #                [4]=mark [5]=is_call [6]=delta [7]=gamma [8]=vega
        #                [9]=theta [10]=spread [11]=div_yield [12]=days_to_div
        #                [13]=momentum_5d [14]=momentum_20d [15]=implied_vol
        # TODO: probably best to either unpack stats array into its 10 parts or forward an entire array for all scalars 
        tau           = torch.tensor(cell_scalars[2],  dtype=torch.float32)
        log_moneyness = torch.tensor(cell_scalars[3],  dtype=torch.float32)
        mark          = torch.tensor(cell_scalars[4],  dtype=torch.float32)
        is_call       = torch.tensor(cell_scalars[5],  dtype=torch.float32)
        ticker        = torch.tensor(ticker_idx,        dtype=torch.long)
        label         = torch.tensor(cell_label,        dtype=torch.float32)
        stats_array   = cell_scalars[6:16] 
        stats         = torch.tensor(stats_array,      dtype=torch.float32)

        # normalize 
        # image         = (image - IMG_MEAN) / (IMG_STD + 1e-8)
        # tau           = (tau - TAU_MEAN) / (TAU_STD + 1e-8)
        # log_moneyness = (log_moneyness - LM_MEAN) / (LM_STD + 1e-8)
        # mark          = (mark - MARK_MEAN) / (MARK_STD + 1e-8)
        # stats         = (stats - STATS_MEAN) / (STATS_STD + 1e-8)

        return image, tau, log_moneyness, is_call, mark, stats, ticker, label
