"""
expand_scalar.py

Utility for loading compressed dataset files and reconstructing the full
120-dim scalar array as it existed before compression.

The compressed format stores scalars as float16 without the OHE ticker block.
This function restores the original float32 array with the OHE appended,
derived from the ticker name in the filename.

Usage:
    from expand_scalar import load_scalars

    scalars = load_scalars('dataset/aapl/2025-01-16_calls.npz')
    # scalars.shape -> (60, 30, 120), dtype float32
"""

import numpy as np
from pathlib import Path
from build_dataset import TICKER_IDX

N_TICKERS      = 104
N_BASE_SCALARS = 16


def load_scalars(npz_path: str | Path) -> np.ndarray:
    """
    Load a compressed .npz scalar file and return the full (60, 30, 120)
    float32 scalar array, identical to the pre-compression format.

    The OHE ticker block is reconstructed from the ticker name in the
    file path (e.g. 'dataset/aapl/...' -> ticker_idx=0).

    Args:
        npz_path : path to <date>_<calls|puts>.npz

    Returns:
        scalars : (HEIGHT, WIDTH, 120) float32
    """
    npz_path = Path(npz_path)

    # Derive ticker from parent directory name
    ticker     = npz_path.parent.name.lower()
    ticker_idx = TICKER_IDX[ticker]

    # Load and upcast base scalars
    data         = np.load(npz_path)
    scalars_base = data["scalars"].astype(np.float32)  # (60, 30, 16)

    # Reconstruct OHE block
    H, W = scalars_base.shape[:2]
    ohe  = np.zeros((H, W, N_TICKERS), dtype=np.float32)
    ohe[:, :, ticker_idx] = 1.0

    return np.concatenate([scalars_base, ohe], axis=-1)  # (60, 30, 120)
