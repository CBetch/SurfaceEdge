import os
import torch
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

# Import your optimized getter and Roberta class
from ClassDefinition.BlueskyLoader import BlueskyGetter

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
TRAIN_DATA_PATH = "./data/master_dataset_train.pt"
TEST_DATA_PATH  = "./data/master_dataset_test.pt"
OUTPUT_PARQUET  = "./data/bluesky_embeddings.parquet"


def get_unique_pairs(*dataset_paths):
    """
    Scans the PyTorch datasets, extracts unique pairs, and dynamically 
    builds the integer-to-string ticker mapping by reading the folder structure.
    """
    unique_pairs = set()
    auto_ticker_map = {}
    
    for path in dataset_paths:
        if not os.path.exists(path):
            print(f"Skipping {path} (Not found)")
            continue
            
        print(f"Scanning {path} for unique dates...")
        data = torch.load(path)
        img_paths = data["img_paths"]
        tickers = data["tickers"]
        
        for img_path, ticker_t in zip(img_paths, tickers):
            # 1. Extract Date from filename
            filename = os.path.basename(img_path)
            date_str = filename.split('_')[0]
            ticker_int = ticker_t.item()
            
            # 2. Extract Ticker from folder name
            if ticker_int not in auto_ticker_map:
                # Normalize slashes (handles both Windows \ and Unix /)
                clean_path = img_path.replace('\\', '/')
                path_parts = clean_path.split('/')
                
                # Grab the parent directory name (e.g., the 'MSFT' in data/MSFT/2025-12-03_calls.png)
                ticker_str = path_parts[-2] 
                
                auto_ticker_map[ticker_int] = ticker_str
                
            unique_pairs.add((date_str, ticker_int))
            
    return unique_pairs, auto_ticker_map

def main():
    # 1. Find all unique pairs
    pairs_to_fetch, TICKER_MAP = get_unique_pairs(TRAIN_DATA_PATH, TEST_DATA_PATH)
    
    print(f"Auto-discovered Ticker Map: {TICKER_MAP}")
    print(f"Found {len(pairs_to_fetch):,} unique (Date, Ticker) pairs to process.")

    if len(pairs_to_fetch) == 0:
        print("No data found. Exiting.")
        return

    # 2. Initialize the Bluesky API
    getter = BlueskyGetter()
    
    results = []
    failed_pairs = []

    # 3. Fetch Data
    print("Beginning API fetches...")
    # Sort pairs so we process chronologically for sanity
    for date_str, ticker_idx in tqdm(sorted(list(pairs_to_fetch)), desc="Fetching Embeddings"):
        
        # ─── NEW: SKIP PRE-BLUESKY DATES ────────────────────
        if date_str < "2023-01-01":
            # Silently skip, or log it if you prefer
            continue
        # ────────────────────────────────────────────────────

        if ticker_idx not in TICKER_MAP:
            print(f"\nWarning: Ticker ID {ticker_idx} not in TICKER_MAP. Skipping.")
            continue

        ticker_str = TICKER_MAP[ticker_idx]
        
        try:
            # Call the optimized fetch function
            avg_emb = getter.fetch_CLS(date=date_str, ticker=ticker_str, limit=5)
            
            if avg_emb is not None:
                # Convert the PyTorch tensor (768,) to a standard Python list for Parquet
                emb_list = avg_emb.detach().cpu().numpy().tolist()
                
                results.append({
                    'date': date_str,
                    'ticker': ticker_idx,  # Save the integer so the Dataset can map it directly
                    'embedding': emb_list
                })
            else:
                failed_pairs.append((date_str, ticker_str))
                
        except Exception as e:
            print(f"\nError fetching {ticker_str} on {date_str}: {e}")
            failed_pairs.append((date_str, ticker_str))

    # 4. Save to Parquet
    print(f"\nSuccessfully fetched {len(results)} embeddings.")
    if failed_pairs:
        print(f"Failed to find posts for {len(failed_pairs)} pairs (These will default to zero-tensors in training).")

    print(f"Saving to {OUTPUT_PARQUET}...")
    df = pd.DataFrame(results)
    
    # Parquet requires strict column types, ensuring 'embedding' is an array of floats
    df.to_parquet(OUTPUT_PARQUET, engine='pyarrow', index=False)
    print("Done!")

if __name__ == "__main__":
    main()
