import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from functools import lru_cache
import os
from datetime import datetime, timedelta
from collections import defaultdict

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


import pandas as pd
from collections import defaultdict

class SequenceSurfaceDataset(Dataset):
    def __init__(self, master_file_path: str, seq_len: int = 25):
        super().__init__()
        print(f"Loading {master_file_path} into RAM...")

        # 1. Load the tabular dataset
        data = torch.load(master_file_path)
        self.scalars   = data["scalars"]    # Shape: [N, 16]
        self.labels    = data["labels"]     # Shape: [N]
        self.tickers   = data["tickers"]    # Shape: [N]
        self.img_paths = data["img_paths"]  # List of strings: length N
        self.seq_len   = seq_len

        print(f"Loaded {len(self.labels):,} flat contract days. Grouping sequences...")
        
        # 2. Build the sequences
        self.valid_sequences = self._build_sequences()
        print(f"Successfully built {len(self.valid_sequences):,} valid {seq_len}-day sequences.")

    def _build_sequences(self):
        contract_groups = defaultdict(list)
        
        # Group by unique contract ID (Unchanged)
        for idx in range(len(self.labels)):
            path = self.img_paths[idx]
            filename = os.path.basename(path)
            date_str = filename.split('_')[0]
            snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")
            
            scalar = self.scalars[idx]
            strike = scalar[1].item()
            tau_days = int(scalar[2].item())
            is_call = scalar[5].item()
            ticker = self.tickers[idx].item()
            
            expiration_date = snapshot_date + timedelta(days=tau_days)
            contract_id = (ticker, is_call, strike, expiration_date)
            contract_groups[contract_id].append((snapshot_date, idx))

        sequences = []
        
        for contract_id, date_idx_list in contract_groups.items():
            date_idx_list.sort(key=lambda x: x[0])
            indices = [idx for _, idx in date_idx_list]
            
            # NEW: Window ends at EVERY day, allowing short histories
            for i in range(len(indices)):
                # Start index goes back up to 24 days, bounded at 0
                start = max(0, i - self.seq_len + 1)
                window = indices[start : i + 1]
                sequences.append(window)
                
        return sequences

    def __len__(self) -> int:
        return len(self.valid_sequences)

    @staticmethod
    @lru_cache(maxsize=2000)
    def load_cached_image(path: str) -> torch.Tensor:
        return read_image(path).float() / 255.0

    def __getitem__(self, idx: int):
        seq_indices = self.valid_sequences[idx]
        actual_len = len(seq_indices)
        pad_len = self.seq_len - actual_len

        # 1. Fetch real data
        images = [self.load_cached_image(self.img_paths[i]) for i in seq_indices]
        images = torch.stack(images) # [actual_len, 3, H, W]
        cell_scalars = self.scalars[seq_indices] # [actual_len, 16]

        # 2. Target Label & Ticker (from the final day of the sequence)
        target_label = self.labels[seq_indices[-1]]
        ticker = self.tickers[seq_indices[-1]]

        # 3. Unpack real scalars
        tau           = cell_scalars[:, 2]
        log_moneyness = cell_scalars[:, 3]
        mark          = cell_scalars[:, 4]
        is_call       = cell_scalars[:, 5]
        stats         = cell_scalars[:, 6:16]

        # 4. Historical Labels (Mask out the target day!)
        hist_labels = self.labels[seq_indices].clone()
        hist_labels[-1] = 0.0 #EXREMELY IMPORTANT  

        # 5. Normalize ONLY the real days
        images        = (images - IMG_MEAN.view(1, 3, 1, 1)) / (IMG_STD.view(1, 3, 1, 1) + 1e-8)
        tau           = (tau - TAU_MEAN) / (TAU_STD + 1e-8)
        log_moneyness = (log_moneyness - LM_MEAN) / (LM_STD + 1e-8)
        mark          = (mark - MARK_MEAN) / (MARK_STD + 1e-8)
        stats         = (stats - STATS_MEAN.unsqueeze(0)) / (STATS_STD.unsqueeze(0) + 1e-8)
        
        hist_labels   = hist_labels.unsqueeze(-1)
        stats         = torch.cat([stats, hist_labels], dim=-1) # [actual_len, 11]

        # 6. Apply Zero-Padding and create the Mask
        padding_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        
        if pad_len > 0:
            # Mask: True means "ignore this position"
            padding_mask[:pad_len] = True
            
            # Pad images
            pad_imgs = torch.zeros(pad_len, 3, images.shape[2], images.shape[3])
            images = torch.cat([pad_imgs, images], dim=0)
            
            # Pad scalars
            tau           = torch.cat([torch.zeros(pad_len), tau], dim=0)
            log_moneyness = torch.cat([torch.zeros(pad_len), log_moneyness], dim=0)
            is_call       = torch.cat([torch.zeros(pad_len), is_call], dim=0)
            mark          = torch.cat([torch.zeros(pad_len), mark], dim=0)
            stats         = torch.cat([torch.zeros(pad_len, 11), stats], dim=0)

        # Notice we are now returning 9 items (added padding_mask)
        return images, tau, log_moneyness, is_call, mark, stats, ticker, target_label, padding_mask


import pandas as pd


class SequenceTextSurfaceDataset(SequenceSurfaceDataset):
    def __init__(self, master_file_path: str, text_data_path: str, seq_len: int = 25):
        # 1. Initialize parent to build sequences, load img_paths, and scalars
        super().__init__(master_file_path=master_file_path, seq_len=seq_len)

        # 2. Load Parquet embeddings into a high-speed lookup dictionary
        self.text_lookup = {}
        if text_data_path and os.path.exists(text_data_path):
            print(f"Loading text embeddings from {text_data_path}...")
            df = pd.read_parquet(text_data_path)
            
            # Map (date_string, ticker_integer) -> 768-dim tensor
            for _, row in df.iterrows():
                d_str = str(row['date'])
                t_int = int(row['ticker'])
                emb_tensor = torch.tensor(row['embedding'], dtype=torch.float32)
                self.text_lookup[(d_str, t_int)] = emb_tensor
                
            print(f"Successfully mapped {len(self.text_lookup)} text embeddings.")
        else:
            raise FileNotFoundError(f"Could not find text data at {text_data_path}")

    def __getitem__(self, idx: int):
        # 1. Fetch the original 9 items from the parent
        # images, tau, log_moneyness, is_call, mark, stats, ticker, target_label, padding_mask
        base_data = super().__getitem__(idx)
        
        # Unpack for clarity
        images, tau, log_moneyness, is_call, mark, stats, ticker, target_label, padding_mask = base_data
        
        # 2. Identify the specific days in this sequence
        seq_indices = self.valid_sequences[idx]
        actual_len = len(seq_indices)
        pad_len = self.seq_len - actual_len
        ticker_val = ticker.item()

        # 3. Retrieve text embeddings for the valid days
        text_seq = []
        for i in seq_indices:
            # Extract date from the path using your original logic (YYYY-MM-DD_...)
            filename = os.path.basename(self.img_paths[i])
            date_str = filename.split('_')[0]
            
            # Get embedding or default to zeros if data was missing for that day
            emb = self.text_lookup.get((date_str, ticker_val), torch.zeros(768))
            text_seq.append(emb)
            
        text_emb_tensor = torch.stack(text_seq) # [actual_len, 768]

        # 4. Apply Zero-Padding to match images and scalars (Pre-padding)
        if pad_len > 0:
            pad_txt = torch.zeros(pad_len, 768)
            text_emb_tensor = torch.cat([pad_txt, text_emb_tensor], dim=0) # [seq_len, 768]

        # 5. Return 10 items in the order expected by run_epoch_text_decoder
        return (
            images,           # [seq_len, 3, H, W]
            tau,              # [seq_len]
            log_moneyness,    # [seq_len]
            is_call,          # [seq_len]
            mark,             # [seq_len]
            stats,            # [seq_len, 11]
            ticker,           # [1]
            text_emb_tensor,  # [seq_len, 768]  <-- NEW
            target_label,     # [1]
            padding_mask      # [seq_len]
        )
