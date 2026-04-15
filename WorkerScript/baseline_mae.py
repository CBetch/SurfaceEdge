"""
baseline.py

Naive baseline evaluation for the SurfaceEdge project.

The naive baseline predicts 0.0 (no price change) for every contract.
MAE under this baseline equals the mean absolute value of all labels,
representing the minimum bar the model must beat to demonstrate that
surface images contain useful predictive information.

Usage:
    from baseline import naive_predict_file, naive_evaluate_dataset

    # Single file
    mae, n = naive_predict_file(
        'dataset/aapl/2025-01-16_calls.npz',
    )

    # Full dataset folder
    mae, n = naive_evaluate_dataset('dataset/aapl')
    mae, n = naive_evaluate_dataset('dataset/aapl', option_type='calls')
    mae, n = naive_evaluate_dataset('dataset/aapl', option_type='puts')
"""

import numpy as np
from pathlib import Path


def naive_predict_file(
    scalar_path: str | Path,
    label_path:  str | Path,
    verbose:     bool = False,
) -> tuple[float, int]:
    """
    Compute naive baseline MAE for a single scalar/label file pair.
    Predicts 0.0 for every non-zero cell.

    Args:
        scalar_path : path to <date>_<calls|puts>_scalars.npy
        label_path  : path to <date>_<calls|puts>_labels.npy
        verbose     : print per-file summary

    Returns:
        mae : mean absolute error for this file
        n   : number of contracts evaluated
    """
    labels = np.load(label_path)
    ys, xs = np.where(labels != 0)

    if len(ys) == 0:
        return 0.0, 0

    actuals = labels[ys, xs].astype(np.float32)
    mae     = float(np.mean(np.abs(actuals)))
    n       = len(actuals)

    if verbose:
        print(f"  Contracts: {n}, Naive MAE: {mae:.6f}")

    return mae, n


def naive_evaluate_dataset(
    dataset_dir: str | Path,
    option_type: str  = "both",
    verbose:     bool = True,
) -> tuple[float, int]:
    """
    Compute naive baseline MAE over all files in a dataset folder.

    Args:
        dataset_dir : path to dataset/<ticker>/ folder
        option_type : 'calls', 'puts', or 'both' (default)
        verbose     : print progress every 200 files

    Returns:
        mae         : mean absolute error across all contracts
        n_contracts : total number of contracts evaluated

    Example:
        from baseline import naive_evaluate_dataset
        mae, n = naive_evaluate_dataset('dataset/aapl')
        mae, n = naive_evaluate_dataset('dataset/aapl', option_type='calls')
    """
    dataset_dir = Path(dataset_dir)
    types = []
    if option_type in ("calls", "both"):
        types.append("calls")
    if option_type in ("puts", "both"):
        types.append("puts")

    all_labels  = []
    files_done  = 0

    for opt_type in types:
        for label_path in sorted(dataset_dir.glob(f"*_{opt_type}_labels.npy")):
            labels = np.load(label_path)
            ys, xs = np.where(labels != 0)
            if len(ys) == 0:
                continue

            all_labels.extend(labels[ys, xs].tolist())
            files_done += 1

            if verbose and files_done % 200 == 0:
                print(f"  {files_done} files, "
                      f"running MAE: {np.mean(np.abs(all_labels)):.6f} "
                      f"({len(all_labels):,} contracts)")

    if not all_labels:
        return 0.0, 0

    all_labels = np.array(all_labels, dtype=np.float32)
    mae = float(np.mean(np.abs(all_labels)))
    n   = len(all_labels)

    if verbose:
        print(f"\n{'─'*45}")
        print(f"Files processed  : {files_done}")
        print(f"Contracts scored : {n:,}")
        print(f"{'─'*45}")
        print(f"{'Naive MAE':<20} {mae:>10.6f}")
        print(f"{'Label std dev':<20} {float(np.std(all_labels)):>10.6f}")
        print(f"{'Label median abs':<20} {float(np.median(np.abs(all_labels))):>10.6f}")
        print(f"{'─'*45}")

    return mae, n


if __name__ == "__main__":
    print("=== CALLS ===")
    naive_evaluate_dataset("dataset/aapl", option_type="calls")
    print()
    print("=== PUTS ===")
    naive_evaluate_dataset("dataset/aapl", option_type="puts")
    print()
    print("=== BOTH ===")
    naive_evaluate_dataset("dataset/aapl", option_type="both")