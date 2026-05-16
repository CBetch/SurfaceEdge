import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

DATASET_DIR = Path("data/processed/filtered")
SPLIT_DATE = "2024-11-21"
TARGET_TICKERS = ["aapl", "msft", "googl", "amzn"]
MIN_NONZERO_CELLS = 100

def compile_split(split="train"):
    print(f"Compiling {split} dataset...")
    ticker_idx_map = {t: i for i, t in enumerate(TARGET_TICKERS)}
    
    all_scalars = []
    all_labels = []
    all_img_paths = []
    all_tickers = []
    
    for ticker in TARGET_TICKERS:
        ticker_dir = DATASET_DIR / ticker
        if not ticker_dir.exists(): continue
            
        for npz_path in tqdm(list(ticker_dir.glob("*_calls.npz")) + list(ticker_dir.glob("*_puts.npz"))):
            date_str = npz_path.stem.split('_')[0]
            
            # Apply chronological split
            if split == "train" and date_str >= SPLIT_DATE: continue
            if split == "test" and date_str < SPLIT_DATE: continue
                
            labels = np.load(npz_path)["labels"]
            ys, xs = np.where(labels != 0)
            if len(ys) < MIN_NONZERO_CELLS: continue
                
            # Load the scalars array we generated in the last step
            scalars = np.load(npz_path.with_name(f"{npz_path.stem}_scalars.npy"))
            png_path = str(npz_path.with_suffix(".png"))
            ticker_idx = ticker_idx_map[ticker]
            
            # Extract valid contracts
            for y, x in zip(ys, xs):
                all_scalars.append(scalars[y, x])
                all_labels.append(labels[y, x])
                all_img_paths.append(png_path)
                all_tickers.append(ticker_idx)

    # Save as a single, highly-optimized PyTorch file
    torch.save({
        "scalars": torch.tensor(np.array(all_scalars), dtype=torch.float32),
        "labels": torch.tensor(np.array(all_labels), dtype=torch.float32),
        "img_paths": all_img_paths,
        "tickers": torch.tensor(all_tickers, dtype=torch.long)
    }, f"master_dataset_{split}.pt")
    print(f"Saved {len(all_labels)} contracts to master_dataset_{split}.pt\n")

if __name__ == "__main__":
    compile_split("train")
    compile_split("test")
