import glob
import os
import torch
import numpy as np
from torch.utils.data import Dataset as D
from torchvision.io import read_image
from functools import lru_cache
from tqdm import tqdm

class InMemoryDataset(D):
    def __init__(self, path, target_ticker=None):
        super().__init__()
        self.samples = []
        self.dates = []
        
        # Make it generic: search all folders, or just the specific ticker folder
        if target_ticker:
            search_pattern = os.path.join(path, target_ticker.lower(), '*_labels.npy')
            print(f"Loading {target_ticker.upper()} dataset into RAM...")
        else:
            search_pattern = os.path.join(path, '*', '*_labels.npy')
            print("Loading FULL dataset into RAM...")
            
        label_files = glob.glob(search_pattern)
        
        for label_file in tqdm(label_files, desc="Parsing Days"):
            base_prefix = label_file.replace('_labels.npy', '')
            date_str = os.path.basename(base_prefix).split('_')[0]
            
            labels_grid = np.load(label_file)
            valid_y, valid_x = np.nonzero(labels_grid)
            
            if len(valid_y) < 100:
                continue
                
            scalars_grid = np.load(f"{base_prefix}_scalars.npy")
            image_tensor = read_image(f"{base_prefix}.png").float() / 255.0
            
            for y, x in zip(valid_y, valid_x):
                cell_scalars = scalars_grid[y, x]
                
                contract_data = {
                    'image': image_tensor, 
                    'tau': torch.tensor(cell_scalars[2], dtype=torch.float32),
                    'log_moneyness': torch.tensor(cell_scalars[3], dtype=torch.float32),
                    'mark': torch.tensor(cell_scalars[4], dtype=torch.float32),
                    'is_call': torch.tensor(cell_scalars[5], dtype=torch.float32),
                    'stats': torch.tensor(cell_scalars[6:16], dtype=torch.float32),
                    'ticker': torch.tensor(np.argmax(cell_scalars[16:120]), dtype=torch.long),
                    'label': torch.tensor(labels_grid[y, x], dtype=torch.float32)
                }
                
                self.samples.append(contract_data)
                self.dates.append(date_str)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = self.samples[idx]
        return d['image'], d['tau'], d['log_moneyness'], d['is_call'], d['mark'], d['stats'], d['ticker'], d['label']

class Dataset(D):
    def __init__(self, path):
        """
        Initializes the dataset by scanning for all label files, filtering out
        sparse days, and building a flat index of valid individual contracts.
        """
        super().__init__()
        self.path = path
        self.samples = [] # Will hold tuples of: (base_file_prefix, y_idx, x_idx)
        
        # 1. Find all label files across all ticker subdirectories
        search_pattern = os.path.join(path, '**', '*_labels.npy')
        label_files = glob.glob(search_pattern, recursive=True)
        
        print(f"Found {len(label_files)} daily options chain grids. Indexing valid contracts...")
        
        # 2. Build the index
        for label_file in label_files:
            # Get the base path (e.g., "dataset/aapl/2020-09-01_calls")
            base_prefix = label_file.replace('_labels.npy', '')
            
            # Load the labels just to find valid coordinates
            labels = np.load(label_file)
            
            # Find indices of non-zero labels (assuming 0.0 means no contract/padding)
            valid_y, valid_x = np.nonzero(labels)
            
            # README Constraint: Skip days with fewer than 100 non-zero cells
            if len(valid_y) < 100:
                continue 
                
            # Add each valid contract to our flat sample list
            for y, x in zip(valid_y, valid_x):
                self.samples.append((base_prefix, y, x))
                
        print(f"Total individual training samples (contracts) indexed: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve the file prefix and grid coordinates for this specific contract
        base_prefix, y, x = self.samples[idx]
        
        # --- 1. Load the Global Surface Image ---
        # torchvision's read_image is fast and returns a tensor of shape [C, H, W]
        image_path = f"{base_prefix}.png"
        image = read_image(image_path).float() / 255.0 
        
        # --- 2. Load Local Scalars and Label ---
        # Using mmap_mode='r' maps the file to memory without loading it all into RAM.
        # This is CRITICAL for training speed when sampling isolated grid cells.
        scalars_path = f"{base_prefix}_scalars.npy"
        labels_path = f"{base_prefix}_labels.npy"
        
        scalars_grid = np.load(scalars_path, mmap_mode='r')
        labels_grid = np.load(labels_path, mmap_mode='r')
        
        # Extract the exact cell (contract) we need
        cell_scalars = scalars_grid[y, x]
        cell_label = labels_grid[y, x]
        
        # --- 3. Parse the 110-dim Scalar Vector ---
        # Based on the README, the scalars are likely ordered as:
        # [0: spot, 1: strike, 2: tau, 3: log_moneyness, 4: mark, 5: is_call, 6-109: ticker_one_hot]
        tau = torch.tensor(cell_scalars[2], dtype=torch.float32)
        log_moneyness = torch.tensor(cell_scalars[3], dtype=torch.float32)
        mark = torch.tensor(cell_scalars[4], dtype=torch.float32)
        is_call = torch.tensor(cell_scalars[5], dtype=torch.float32)
        
        # The training loop will handle converting this 1-hot vector into an integer 
        # index for the model's Embedding layer.
        ticker_one_hot = torch.tensor(cell_scalars[6:110], dtype=torch.float32) 
        
        label = torch.tensor(cell_label, dtype=torch.float32)
        
        # Return exactly what the train_model loop expects to unpack
        return image, tau, log_moneyness, is_call, mark, ticker_one_hot, label
    
class FastDataset(Dataset):
    def __init__(self, path):
        super().__init__(path)
    
    # Each DataLoader worker will cache up to 64 unique days in its own RAM.
    # If it sees a prefix it has loaded recently, it skips the hard drive entirely.
    @lru_cache(maxsize=64) 
    def _get_day_data(self, base_prefix):
        # Read the image once
        image = read_image(f"{base_prefix}.png").float() / 255.0 
        
        # We drop mmap_mode='r' here. Loading it fully into RAM once is 
        # much faster than constantly querying the disk via mmap.
        scalars_grid = np.load(f"{base_prefix}_scalars.npy")
        labels_grid = np.load(f"{base_prefix}_labels.npy")
        
        return image, scalars_grid, labels_grid

    def __getitem__(self, idx):
        base_prefix, y, x = self.samples[idx]
        
        # 1. Pull the whole day's grid from RAM (or disk if it's not cached yet)
        image, scalars_grid, labels_grid = self._get_day_data(base_prefix)
        
        # 2. Extract the exact cell (contract)
        cell_scalars = scalars_grid[y, x]
        cell_label = labels_grid[y, x]
        
        # 3. Parse the 110-dim Scalar Vector (Same as your original logic)
        tau = torch.tensor(cell_scalars[2], dtype=torch.float32)
        log_moneyness = torch.tensor(cell_scalars[3], dtype=torch.float32)
        mark = torch.tensor(cell_scalars[4], dtype=torch.float32)
        is_call = torch.tensor(cell_scalars[5], dtype=torch.float32)
        ticker_one_hot = torch.tensor(cell_scalars[6:110], dtype=torch.float32) 
        label = torch.tensor(cell_label, dtype=torch.float32)
        
        return image, tau, log_moneyness, is_call, mark, ticker_one_hot, label