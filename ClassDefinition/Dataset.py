import glob
import os
import torch
import numpy as np
from torch.utils.data import Dataset as D
from torchvision.io import read_image

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