import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm


SEROOT = os.environ.get('SEROOT', '.')
sys.path.append(SEROOT)
from ClassDefinition.Model import SurfaceEdgeModel
from ClassDefinition.Dataset import Dataset , FastDataset, InMemoryDataset

from functools import lru_cache



def train_model(model, train_loader, val_loader, device, epochs=15, lr=5e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.L1Loss() 
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_baseline = 0.0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_bar = tqdm(train_loader, desc="Training")
        
        for batch in train_bar:
            # Unpack the 8 dataset yields
            image, tau, log_moneyness, is_call, mark, stats, ticker, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(image, tau, log_moneyness, is_call, mark, stats, ticker)
            predictions = predictions.squeeze(-1) 
            
            loss = criterion(predictions, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            naive_preds = torch.zeros_like(labels)
            baseline_loss = criterion(naive_preds, labels).item()
            train_baseline += baseline_loss
            
            train_bar.set_postfix({
                'MAE': f"{loss.item():.4f}", 
                'Base_MAE': f"{baseline_loss:.4f}"
            })
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_base = train_baseline / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_baseline = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                image, tau, log_moneyness, is_call, mark, stats, ticker, labels = [b.to(device) for b in batch]
                
                predictions = model(image, tau, log_moneyness, is_call, mark, stats, ticker).squeeze(-1)
                
                val_loss += criterion(predictions, labels).item()
                val_baseline += criterion(torch.zeros_like(labels), labels).item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_base = val_baseline / len(val_loader)
        
        print(f"Train MAE: {avg_train_loss:.4f} (Baseline: {avg_train_base:.4f})")
        print(f"Val MAE:   {avg_val_loss:.4f} (Baseline: {avg_val_base:.4f})")
        
        if avg_val_loss < avg_val_base:
            torch.save(model.state_dict(), f"surface_edge_epoch_{epoch+1}.pt")
            print(f"[*] Checkpoint saved! Model is beating the baseline by {(avg_val_base - avg_val_loss):.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = SurfaceEdgeModel().to(device)
    
    # Target only MSFT to keep RAM usage sane
    full_dataset = InMemoryDataset(path=os.path.join(SEROOT, 'dataset'), target_ticker='msft')
    
    # Chronological Split
    unique_dates = sorted(list(set(full_dataset.dates)))
    split_idx = int(0.8 * len(unique_dates))
    train_dates = set(unique_dates[:split_idx])
    
    train_indices = [i for i, d in enumerate(full_dataset.dates) if d in train_dates]
    val_indices = [i for i, d in enumerate(full_dataset.dates) if d not in train_dates]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Total Unique Days: {len(unique_dates)} (Train: {split_idx}, Val: {len(unique_dates) - split_idx})")
    print(f"Training samples: {len(train_dataset):,} | Validation samples: {len(val_dataset):,}")
    
    # num_workers=0 because it's loaded in RAM. Batch size is safe to crank up here.
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    
    train_model(model, train_loader, val_loader, device, epochs=10, lr=5e-4)

if __name__ == '__main__':
    main()