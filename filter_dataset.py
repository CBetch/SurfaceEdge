"""
filter_dataset.py

Filters the raw dataset into dataset_filtered by removing outlier contracts:
    - abs(label) > 2.0  : contracts moving more than 200% overnight
    - spread > 0.5      : contracts where bid-ask gap > 50% of mark price

Original dataset is untouched. Filtered dataset is written to dataset_filtered/.
Days with fewer than 100 valid contracts after filtering are dropped entirely.

Usage:
    from filter_dataset import filter_all
    filter_all(dataset_dir, filtered_dir)

    # Or run directly:
    python filter_dataset.py

EX on Apple:
=== AAPL Filter Results ===
Before : 484,011 contracts
  MAE    : 0.359672
  Std dev: 3.319277
  Median : 0.073132

After  : 439,593 contracts (90.8% retained)
  MAE    : 0.136049
  Std dev: 0.235743
  Median : 0.060713

Days dropped (originally sparse): 0
"""

import numpy as np
import shutil
from pathlib import Path

MAX_LABEL   = 2.0
MAX_SPREAD  = 0.5
MIN_NONZERO = 100


def filter_ticker(ticker_src: Path, ticker_dst: Path) -> tuple[list, list, int]:
    """
    Filter one ticker's dataset into ticker_dst.
    Returns (before_labels, after_labels, files_deleted).
    """
    ticker_dst.mkdir(parents=True, exist_ok=True)

    before_labels = []
    after_labels  = []
    files_deleted = 0

    for npz_path in sorted(ticker_src.glob('*.npz')):
        data    = np.load(npz_path)
        labels  = data['labels'].copy()
        scalars = data['scalars']

        ys, xs = np.where(labels != 0)
        original_count = len(ys)

        for y, x in zip(ys, xs):
            label  = float(labels[y, x])
            spread = float(scalars[y, x, 10])
            before_labels.append(label)
            if abs(label) > MAX_LABEL or spread > MAX_SPREAD:
                labels[y, x] = 0.0

        # Only drop the day if it was already sparse before filtering
        if original_count < MIN_NONZERO:
            files_deleted += 1
            continue

        np.savez_compressed(ticker_dst / npz_path.name,
                            labels=labels, scalars=scalars)

        src_png = npz_path.with_suffix('.png')
        if src_png.exists():
            shutil.copy2(src_png, ticker_dst / src_png.name)

        ys2, xs2 = np.where(labels != 0)
        after_labels.extend(labels[ys2, xs2].tolist())

    return before_labels, after_labels, files_deleted


def filter_all(dataset_dir: str | Path, filtered_dir: str | Path) -> None:
    """
    Filter all tickers from dataset_dir into filtered_dir.
    Reports per-ticker and overall MAE before/after.
    """
    dataset_dir = Path(dataset_dir)
    filtered_dir = Path(filtered_dir)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    tickers = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])

    grand_before        = []
    grand_after         = []
    total_days_dropped  = 0

    for ticker in tickers:
        before, after, dropped = filter_ticker(
            dataset_dir  / ticker,
            filtered_dir / ticker,
        )

        b = np.array(before)
        a = np.array(after)
        retained = len(a) / len(b) * 100 if len(b) > 0 else 0

        print(f'{ticker:>6} | '
              f'Before: {len(b):>7,} MAE={np.mean(np.abs(b)):.4f} | '
              f'After: {len(a):>7,} ({retained:.1f}%) MAE={np.mean(np.abs(a)):.4f} | '
              f'Dropped {dropped} days')

        grand_before.extend(before)
        grand_after.extend(after)
        total_days_dropped += dropped

    gb = np.array(grand_before)
    ga = np.array(grand_after)

    print()
    print('─' * 80)
    print(f'TOTAL | Before: {len(gb):,} contracts  MAE={np.mean(np.abs(gb)):.6f}')
    print(f'TOTAL | After : {len(ga):,} contracts  MAE={np.mean(np.abs(ga)):.6f}')
    print(f'Contracts removed : {len(gb)-len(ga):,} ({100*(len(gb)-len(ga))/len(gb):.1f}%)')
    print(f'Days dropped      : {total_days_dropped:,}')
    print(f'Filtered dataset  : {filtered_dir}')


if __name__ == '__main__':
    import sys
    dataset_dir  = sys.argv[1] if len(sys.argv) > 1 else 'dataset'
    filtered_dir = sys.argv[2] if len(sys.argv) > 2 else 'dataset_filtered'
    filter_all(dataset_dir, filtered_dir)