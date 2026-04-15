"""
main.py

Training entry point for SurfaceEdge.

Runs one full train epoch followed by test set evaluation.
Reports MAE for both splits and compares against the naive 0.0 baseline.

Usage:
    python main.py
    python main.py --dataset dataset --epochs 1 --batch_size 256 --lr 1e-4
"""
import os
import argparse
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

SEROOT = os.environ.get('SEROOT', '.')
sys.path.append(SEROOT)
from ClassDefinition.Model import SurfaceEdgeModel
from ClassDefinition.SurfaceDataset import SurfaceDataset


# ── Config ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SurfaceEdge Training")
    parser.add_argument("--dataset",     type=str,   default=f"{SEROOT}/data/processed/filtered")
    parser.add_argument("--split_date",  type=str,   default="2024-11-21")
    # option_type is always 'both' — trains on calls and puts simultaneously
    parser.add_argument("--option_type", type=str,   default="both")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--save_path",   type=str,   default=f"{SEROOT}/Artifacts/surfaceedge")
    return parser.parse_args()


# ── Training ──────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train: bool) -> tuple[float, int]:
    """
    Run one epoch. Returns (mae, n_samples).
    If train=True, runs backward pass and optimizer step.
    """
    model.train(train)
    total_loss = 0.0
    total_n    = 0

    desc = "Train" if train else "Test "
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
            # image, tau, log_moneyness, is_call, mark, ticker, label = [
            #    b.to(device) for b in batch
            # ]
            image, tau, log_moneyness, is_call, mark, stats, ticker, label = [b.to(device) for b in batch]

            preds = model(
                image         = image,
                tau           = tau,
                log_moneyness = log_moneyness,
                is_call       = is_call,
                mark          = mark,
                stats         = stats, 
                ticker        = ticker,
            ).squeeze(1)  # (B,)

            loss = torch.mean(torch.abs(preds - label))  # MAE

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(label)
            total_n    += len(label)

    return total_loss / total_n, total_n


def naive_mae(dataset) -> float:
    """Compute naive baseline MAE directly from the contract index — no file loading."""
    import numpy as np
    from pathlib import Path

    total_abs = 0.0
    total_n   = 0

    # Group samples by npz file to minimize loads
    from collections import defaultdict
    file_groups = defaultdict(list)
    for npz_path_str, y, x, _ in dataset.samples:
        file_groups[npz_path_str].append((y, x))

    for npz_path_str, coords in file_groups.items():
        data   = np.load(npz_path_str)
        labels = data["labels"]
        for y, x in coords:
            total_abs += abs(float(labels[y, x]))
            total_n   += 1

    return total_abs / total_n if total_n > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Dataset     : {args.dataset}")
    print(f"Split date  : {args.split_date}")
    print(f"Option type : both")
    print(f"Batch size  : {args.batch_size}")
    print(f"LR          : {args.lr}")
    print()

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = SurfaceDataset(args.dataset, split="train",
                               split_date=args.split_date,
                               option_type='both', ticker_list=["aapl", "msft", "googl", "amzn"])
    test_ds  = SurfaceDataset(args.dataset, split="test",
                               split_date=args.split_date,
                               option_type='both', ticker_list=["aapl", "msft", "googl", "amzn"]) #TODO option_type use gargs 

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
        persistent_workers = args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
        persistent_workers = args.num_workers > 0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SurfaceEdgeModel().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Naive baseline ────────────────────────────────────────────────────────
    print("\nComputing naive baseline MAE on test set ...")
    naive = naive_mae(test_ds)
    print(f"  Naive MAE (test) : {naive:.6f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_mae, train_n = run_epoch(model, train_loader, optimizer, device, train=True)
        test_mae,  test_n  = run_epoch(model, test_loader,  optimizer, device, train=False)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:>3}/{args.epochs} | "
              f"Train MAE: {train_mae:.6f} ({train_n:,} contracts) | "
              f"Test MAE:  {test_mae:.6f} ({test_n:,} contracts) | "
              f"Naive: {naive:.6f} | "
              f"Beat naive: {'YES' if test_mae < naive else 'NO'} | "
              f"{elapsed:.1f}s")

    # ── Save model ────────────────────────────────────────────────────────────
        save_path = f"{args.save_path}_epoch{epoch}"
        torch.save({
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epochs_trained":       args.epochs,
            "final_train_mae":      train_mae,
            "final_test_mae":       test_mae,
            "naive_mae":            naive,
            "split_date":           args.split_date,
            "option_type":          args.option_type,
        }, save_path)
        print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
