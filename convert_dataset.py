import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Update this to match your actual dataset path
DATASET_DIR = Path("data/processed/filtered") 

# Only process these specific subdirectories
TARGET_TICKERS = {"aapl", "msft", "googl", "amzn"}

def convert_npz_to_npy(npz_path: Path):
    try:
        data = np.load(npz_path)
        
        scalars_path = npz_path.with_name(f"{npz_path.stem}_scalars.npy")
        labels_path  = npz_path.with_name(f"{npz_path.stem}_labels.npy")
        
        np.save(scalars_path, data["scalars"].astype(np.float32))
        np.save(labels_path, data["labels"])
        
    except Exception as e:
        print(f"Error processing {npz_path}: {e}")

def main():
    print(f"Scanning {DATASET_DIR} for {TARGET_TICKERS}...")
    
    npz_files = []
    # Iterate through the main directory
    for ticker_dir in DATASET_DIR.iterdir():
        # Only enter the directory if it matches one of our target tickers
        if ticker_dir.is_dir() and ticker_dir.name.lower() in TARGET_TICKERS:
            npz_files.extend(list(ticker_dir.glob("*.npz")))
            
    print(f"Found {len(npz_files):,} files to convert.")
    
    if not npz_files:
        print("No files found. Please check your DATASET_DIR path.")
        return

    # Process only the filtered files
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(
            executor.map(convert_npz_to_npy, npz_files), 
            total=len(npz_files), 
            desc="Converting to .npy"
        ))
        
    print("\nConversion complete for target tickers!")

if __name__ == "__main__":
    main()
