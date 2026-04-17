"""
compute_dataset_stats.py

Standalone script to calculate the exact global means and standard deviations
for the continuous inputs of the SurfaceEdge dataset, isolated to the training split.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set up repository root and path
SEROOT = os.environ.get('SEROOT', '.')
sys.path.append(SEROOT)

from ClassDefinition.SurfaceDataset import SurfaceDataset

def compute_dataset_statistics(dataset_dir=f"{SEROOT}/data/processed/filtered", batch_size=256, num_workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset directory: {dataset_dir}")

    # 1. Load ONLY the training split for the 4 specific tickers
    train_ds = SurfaceDataset(
        dataset_dir, 
        split='train',
        ticker_list=["aapl", "msft", "googl", "amzn"]
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    # 2. Initialize float64 running accumulators for precision
    # Image (3 channels)
    img_sum = torch.zeros(3, dtype=torch.float64, device=device)
    img_sq_sum = torch.zeros(3, dtype=torch.float64, device=device)
    img_pixel_count = 0

    # Individual Scalars
    tau_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
    tau_sq_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
    
    lm_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
    lm_sq_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
    
    mark_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
    mark_sq_sum = torch.tensor(0.0, dtype=torch.float64, device=device)

    # Stats Array (10 features)
    stats_sum = torch.zeros(10, dtype=torch.float64, device=device)
    stats_sq_sum = torch.zeros(10, dtype=torch.float64, device=device)
    
    scalar_count = 0

    # 3. Iterate through the dataset
    print(f"Iterating through {len(train_ds)} training samples to compute statistics...")
    for batch in tqdm(train_loader):
        # Unpack exactly what your __getitem__ returns
        image, tau, log_moneyness, is_call, mark, stats, ticker, label = batch
        
        # Move to device
        image = image.to(device).double()
        tau = tau.to(device).double()
        log_moneyness = log_moneyness.to(device).double()
        mark = mark.to(device).double()
        stats = stats.to(device).double()

        # --- Accumulate Image Stats ---
        img_sum += image.sum(dim=[0, 2, 3])
        img_sq_sum += (image ** 2).sum(dim=[0, 2, 3])
        img_pixel_count += image.shape[0] * image.shape[2] * image.shape[3]

        # --- Accumulate Scalar Stats ---
        scalar_count += tau.shape[0]

        tau_sum += tau.sum()
        tau_sq_sum += (tau ** 2).sum()

        lm_sum += log_moneyness.sum()
        lm_sq_sum += (log_moneyness ** 2).sum()

        mark_sum += mark.sum()
        mark_sq_sum += (mark ** 2).sum()

        stats_sum += stats.sum(dim=0)
        stats_sq_sum += (stats ** 2).sum(dim=0)

    # 4. Compute Final Means and Standard Deviations
    print("\n" + "="*40)
    print("FINAL NORMALIZATION VALUES")
    print("="*40)

    # Image
    img_mean = img_sum / img_pixel_count
    img_var = (img_sq_sum / img_pixel_count) - (img_mean ** 2)
    img_std = torch.sqrt(img_var)
    print("\n--- Image (Shape: 3) ---")
    print(f"Mean: {img_mean.cpu().float().tolist()}")
    print(f"Std:  {img_std.cpu().float().tolist()}")

    # Scalars
    tau_mean = tau_sum / scalar_count
    tau_std = torch.sqrt((tau_sq_sum / scalar_count) - (tau_mean ** 2))
    
    lm_mean = lm_sum / scalar_count
    lm_std = torch.sqrt((lm_sq_sum / scalar_count) - (lm_mean ** 2))

    mark_mean = mark_sum / scalar_count
    mark_std = torch.sqrt((mark_sq_sum / scalar_count) - (mark_mean ** 2))

    stats_mean = stats_sum / scalar_count
    stats_var = (stats_sq_sum / scalar_count) - (stats_mean ** 2)
    stats_std = torch.sqrt(stats_var)

    print("\n--- Individual Scalars ---")
    print(f"Tau Mean: {tau_mean.item():.6f} | Std: {tau_std.item():.6f}")
    print(f"Log Moneyness Mean: {lm_mean.item():.6f} | Std: {lm_std.item():.6f}")
    print(f"Mark Mean: {mark_mean.item():.6f} | Std: {mark_std.item():.6f}")

    print("\n--- Stats Array (Shape: 10) ---")
    print(f"Mean: {stats_mean.cpu().float().tolist()}")
    print(f"Std:  {stats_std.cpu().float().tolist()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Normalization Statistics")
    parser.add_argument("--dataset", type=str, default=f"{SEROOT}/data/processed/filtered")
    args = parser.parse_args()
    
    compute_dataset_statistics(dataset_dir=args.dataset)
