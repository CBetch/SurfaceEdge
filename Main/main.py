"""
main.py

Training entry point for SurfaceEdge.

Runs one full train epoch followed by test set evaluation.
Reports MAE for both splits and compares against the naive 0.0 baseline.
"""
import os
import argparse
import time
import random  # <-- NEW: Required for random search
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
    parser.add_argument("--train_data",  type=str,   default=f"{SEROOT}/data/master_dataset_train.pt")
    parser.add_argument("--test_data",   type=str,   default=f"{SEROOT}/data/master_dataset_test.pt")
    parser.add_argument("--option_type", type=str,   default="both")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--save_path",   type=str,   default=f"{SEROOT}/Artifacts/surfaceedge")
    return parser.parse_args()

# ── Training ──────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train: bool) -> tuple[float, int]:
    model.train(train)
    total_loss = 0.0
    total_n    = 0

    desc = "Train" if train else "Test "
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
            image, tau, log_moneyness, is_call, mark, stats, ticker, label = [b.to(device) for b in batch]

            preds = model(
                image         = image,
                tau           = tau,
                log_moneyness = log_moneyness,
                is_call       = is_call,
                mark          = mark,
                stats         = stats,
                ticker        = ticker,
            ).squeeze(1)

            loss = torch.mean(torch.abs(preds - label))

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(label)
            total_n    += len(label)

    return total_loss / total_n, total_n

def naive_mae(dataset) -> float:
    """Compute naive baseline MAE instantly from the pre-loaded tensor in RAM."""
    if len(dataset.labels) == 0:
        return 0.0
    return torch.mean(torch.abs(dataset.labels)).item()

# ── Main ──────────────────────────────────────────────────────────────────────

def hyper_search():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Train Data  : {args.train_data}")
    print(f"Test Data   : {args.test_data}")
    print()

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = SurfaceDataset(master_file_path=args.train_data)
    test_ds  = SurfaceDataset(master_file_path=args.test_data)

    # ── Naive baseline ────────────────────────────────────────────────────────
    print("\nComputing naive baseline MAE on test set ...")
    naive = naive_mae(test_ds)
    print(f"  Naive MAE (test) : {naive:.6f}\n")

    # ── Hyperparameter Search Space ───────────────────────────────────────────
    learning_rates = [.0005, .0001, .00005, .00001]
    dropouts       = [.2, .15, .1, .05]
    batch_sizes    = [512, 256, 128, 64]
    hidden_dims    = [128, 64, 32]
    img_dims       = [256, 128, 64]
    ticker_dims    = [128, 64, 32]

    # Tracker for the best model across all iterations
    best_overall_mae = float('inf')
    best_config = {}
    iteration = 0

    while True:
        iteration += 1

        # 1. Randomly sample hyperparameters
        current_lr         = random.choice(learning_rates)
        current_dropout    = random.choice(dropouts)
        current_batch_size = random.choice(batch_sizes)
        current_hidden_dim = random.choice(hidden_dims)
        current_img_dim    = random.choice(img_dims)
        current_ticker_dim = random.choice(ticker_dims)

        print(f"\n{'='*70}")
        print(f"ITERATION {iteration} HYPERPARAMETERS")
        print(f"{'='*70}")
        print(f"LR         : {current_lr}")
        print(f"Batch Size : {current_batch_size}")
        print(f"Dropout    : {current_dropout}")
        print(f"Hidden Dim : {current_hidden_dim}")
        print(f"Img Dim    : {current_img_dim}")
        print(f"Ticker Dim : {current_ticker_dim}")
        print(f"{'-'*70}")

        # 2. Inject current_batch_size into DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size  = current_batch_size,
            shuffle     = True,
            num_workers = args.num_workers,
            prefetch_factor = 2 if args.num_workers > 0 else None,
            pin_memory  = device.type == "cuda",
            persistent_workers = args.num_workers > 0,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size  = current_batch_size,
            shuffle     = False,
            num_workers = args.num_workers,
            prefetch_factor = 2 if args.num_workers > 0 else None,
            pin_memory  = device.type == "cuda",
            persistent_workers = args.num_workers > 0,
        )

        # 3. Inject dimensions into the Model
        model = SurfaceEdgeModel(
            img_dim    = current_img_dim,
            ticker_dim = current_ticker_dim,
            dropout    = current_dropout,
            hidden_dim = current_hidden_dim
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model params: {n_params:,}")

        # 4. Inject current_lr into Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)

        # ── Training loop ─────────────────────────────────────────────────────
        print()
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            train_mae, train_n = run_epoch(model, train_loader, optimizer, device, train=True)
            test_mae,  test_n  = run_epoch(model, test_loader,  optimizer, device, train=False)

            elapsed = time.time() - t0
            
            # Check if this is the best model so far
            is_best = False
            if test_mae < best_overall_mae:
                best_overall_mae = test_mae
                is_best = True
                best_config = {
                    'iteration': iteration,
                    'epoch': epoch,
                    'mae': test_mae,
                    'lr': current_lr,
                    'batch': current_batch_size,
                    'drop': current_dropout,
                    'hidden': current_hidden_dim,
                    'img': current_img_dim,
                    'ticker': current_ticker_dim
                }

            best_tag = "[NEW BEST!]" if is_best else ""
            
            print(f"Epoch {epoch:>3}/{args.epochs} "
                  f"Train MAE: {train_mae:.6f} "
                  f"Test MAE:  {test_mae:.6f} "
                  f"Naive: {naive:.6f} "
                  f"{elapsed:.1f}s {best_tag}")

            # ── Save model ────────────────────────────────────────────────────
            save_path = f"{args.save_path}_iter{iteration}_epoch{epoch}"
            torch.save({
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epochs_trained":       args.epochs,
                "final_train_mae":      train_mae,
                "final_test_mae":       test_mae,
                "naive_mae":            naive,
                "option_type":          args.option_type,
                # Save the hyperparameters inside the checkpoint too!
                "hyperparameters": {
                    "lr": current_lr,
                    "batch_size": current_batch_size,
                    "dropout": current_dropout,
                    "hidden_dim": current_hidden_dim,
                    "img_dim": current_img_dim,
                    "ticker_dim": current_ticker_dim
                }
            }, save_path)
        
        # Print the reigning champion at the end of every iteration
        print(f"\nBEST OVERALL MODEL SO FAR (Iteration {best_config['iteration']}, Epoch {best_config['epoch']}):")
        print(f"   Test MAE: {best_config['mae']:.6f}")
        print(f"   Config:   LR={best_config['lr']}, Batch={best_config['batch']}, Drop={best_config['drop']}, "
              f"Hidden={best_config['hidden']}, Img={best_config['img']}, Ticker={best_config['ticker']}")

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Train Data  : {args.train_data}")
    print(f"Test Data   : {args.test_data}")
    print()

    # ── Fixed Hyperparameters ─────────────────────────────────────────────────
    lr         = 1e-05
    batch_size = 256
    dropout    = 0.2
    hidden_dim = 128
    img_dim    = 64
    ticker_dim = 128

    print(f"{'='*50}")
    print("TRAINING WITH FIXED HYPERPARAMETERS")
    print(f"{'='*50}")
    print(f"LR         : {lr}")
    print(f"Batch Size : {batch_size}")
    print(f"Dropout    : {dropout}")
    print(f"Hidden Dim : {hidden_dim}")
    print(f"Img Dim    : {img_dim}")
    print(f"Ticker Dim : {ticker_dim}")
    print(f"{'-'*50}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = SurfaceDataset(master_file_path=args.train_data)
    test_ds  = SurfaceDataset(master_file_path=args.test_data)

    # ── Naive baseline ────────────────────────────────────────────────────────
    print("Computing naive baseline MAE on test set ...")
    naive = naive_mae(test_ds)
    print(f"  Naive MAE (test) : {naive:.6f}\n")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        prefetch_factor = 2 if args.num_workers > 0 else None,
        pin_memory  = device.type == "cuda",
        persistent_workers = args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        prefetch_factor = 2 if args.num_workers > 0 else None,
        pin_memory  = device.type == "cuda",
        persistent_workers = args.num_workers > 0,
    )

    # ── Model & Optimizer ─────────────────────────────────────────────────────
    model = SurfaceEdgeModel(
        img_dim    = img_dim,
        ticker_dim = ticker_dim,
        dropout    = dropout,
        hidden_dim = hidden_dim
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_test_mae = float('inf')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_mae, train_n = run_epoch(model, train_loader, optimizer, device, train=True)
        test_mae,  test_n  = run_epoch(model, test_loader,  optimizer, device, train=False)

        elapsed = time.time() - t0

        # Track the best model
        is_best = False
        if test_mae < best_test_mae:
            best_test_mae = test_mae
            is_best = True

        best_tag = "[NEW BEST!]" if is_best else ""

        print(f"Epoch {epoch:>3}/{args.epochs} "
              f"Train MAE: {train_mae:.6f} "
              f"Test MAE:  {test_mae:.6f} "
              f"Naive: {naive:.6f} "
              f"{elapsed:.1f}s {best_tag}")

        # ── Save model ────────────────────────────────────────────────────────
        save_path = f"{args.save_path}_epoch{epoch}.pt"
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epochs_trained":       args.epochs,
            "final_train_mae":      train_mae,
            "final_test_mae":       test_mae,
            "naive_mae":            naive,
            "option_type":          args.option_type,
            "hyperparameters": {
                "lr":         lr,
                "batch_size": batch_size,
                "dropout":    dropout,
                "hidden_dim": hidden_dim,
                "img_dim":    img_dim,
                "ticker_dim": ticker_dim
            }
        }, save_path)

    print(f"\nTraining Complete. Best Test MAE: {best_test_mae:.6f}")


if __name__ == "__main__":
    main()
