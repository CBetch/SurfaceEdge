import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

SEROOT = os.environ.get('SEROOT', '.')
sys.path.append(SEROOT)
from ClassDefinition.Model import SurfaceEdgeModel
from ClassDefinition.Dataset import Dataset 

def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.L1Loss() # MAE loss aligns with the README baseline requirement
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_baseline = 0.0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_bar = tqdm(train_loader, desc="Training")
        
        for batch in train_bar:
            # Unpack the dataset yields
            image, tau, log_moneyness, is_call, mark, ticker_raw, labels = [b.to(device) for b in batch]
            
            # Convert 1-hot ticker (from scalars) to index
            if ticker_raw.dim() > 1 and ticker_raw.shape[1] > 1:
                ticker = torch.argmax(ticker_raw, dim=1)
            else:
                ticker = ticker_raw.long()

            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(image, tau, log_moneyness, is_call, mark, ticker)
            
            # Squeeze output from [Batch, 1] to [Batch] to match labels
            predictions = predictions.squeeze(-1) 
            
            # Calculate Loss (MAE)
            loss = criterion(predictions, labels)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            
            # Tracking
            train_loss += loss.item()
            
            # Calculate naive baseline MAE (predicting 0.0 price change)
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
                image, tau, log_moneyness, is_call, mark, ticker_raw, labels = [b.to(device) for b in batch]
                
                if ticker_raw.dim() > 1 and ticker_raw.shape[1] > 1:
                    ticker = torch.argmax(ticker_raw, dim=1)
                else:
                    ticker = ticker_raw.long()

                predictions = model(image, tau, log_moneyness, is_call, mark, ticker).squeeze(-1)
                
                val_loss += criterion(predictions, labels).item()
                val_baseline += criterion(torch.zeros_like(labels), labels).item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_base = val_baseline / len(val_loader)
        
        print(f"Train MAE: {avg_train_loss:.4f} (Baseline: {avg_train_base:.4f})")
        print(f"Val MAE:   {avg_val_loss:.4f} (Baseline: {avg_val_base:.4f})")
        
        # Save checkpoint if it beats the baseline
        if avg_val_loss < avg_val_base:
            torch.save(model.state_dict(), f"surface_edge_epoch_{epoch+1}.pt")
            print(f"[*] Checkpoint saved! Model is beating the baseline by {(avg_val_base - avg_val_loss):.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = SurfaceEdgeModel().to(device)
    
    # 2. Initialize Dataset
    # __getitme__ returns exactly 7 items in this order:
    # (image, tau, log_moneyness, is_call, mark, ticker, label)
    print("Loading dataset...")
    # full_dataset = Dataset(data_dir=os.path.join(SEROOT, 'dataset')) 
    full_dataset = Dataset(path=os.path.join(SEROOT, 'dataset'))
    
    # 80/20 train/dev 
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training samples: {train_size:,} | Validation samples: {val_size:,}")
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    #Start Training
    train_model(model, train_loader, val_loader, device, epochs=15, lr=5e-4)

if __name__ == '__main__':
    main()
