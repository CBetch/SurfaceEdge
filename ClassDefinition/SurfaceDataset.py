import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from functools import lru_cache


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


class SurfaceDataset(Dataset):
    def __init__(self, master_file_path: str):
        super().__init__()
        print(f"Loading {master_file_path} into RAM...")
        
        # Load the entire tabular dataset directly into memory
        data = torch.load(master_file_path)
        self.scalars   = data["scalars"]    # Shape: [N, 16]
        self.labels    = data["labels"]     # Shape: [N]
        self.tickers   = data["tickers"]    # Shape: [N]
        self.img_paths = data["img_paths"]  # List of strings: length N
        
        print(f"Loaded {len(self.labels):,} contracts.")

    def __len__(self) -> int:
        return len(self.labels)

    # Cache the images so we don't re-read the same PNG for contracts on the same day
    @staticmethod
    @lru_cache(maxsize=2000) 
    def load_cached_image(path: str) -> torch.Tensor:
        return read_image(path).float() / 255.0

    def __getitem__(self, idx: int):
        # 1. Image (Cached read)
        image = self.load_cached_image(self.img_paths[idx])
        
        # 2. Instant memory lookup for tabular data
        cell_scalars = self.scalars[idx]
        label        = self.labels[idx]
        ticker       = self.tickers[idx]

        # 3. Unpack scalars
        tau           = cell_scalars[2]
        log_moneyness = cell_scalars[3]
        mark          = cell_scalars[4]
        is_call       = cell_scalars[5]
        stats         = cell_scalars[6:16] 

    
        # normalize 
        image         = (image - IMG_MEAN) / (IMG_STD + 1e-8)
        tau           = (tau - TAU_MEAN) / (TAU_STD + 1e-8)
        log_moneyness = (log_moneyness - LM_MEAN) / (LM_STD + 1e-8)
        mark          = (mark - MARK_MEAN) / (MARK_STD + 1e-8)
        stats         = (stats - STATS_MEAN) / (STATS_STD + 1e-8)

        return image, tau, log_moneyness, is_call, mark, stats, ticker, label
