"""
download_data.py

Downloads the full options + underlying dataset from philippdubach/options-data.
Roughly 9.4 GB on Disk
Each ticker gets two parquet files saved under:

    data/
        <ticker>/
            options.parquet
            underlying.parquet

Run once before any other scripts:
    python download_data.py
"""

import urllib.request
from pathlib import Path
import os 
SEROOT = os.environ.get('SEROOT', '.')

# Cloudflare blocks Python's default user-agent — spoof a browser
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}

BASE_URL = "https://static.philippdubach.com/data/options"

DATA_DIR = Path(SEROOT) / "data" / "unprocessed"

TICKERS = [
    "aapl", "abbv", "abt",   "acn",   "adbe", "aig",  "amd",  "amgn", "amt",  "amzn",
    "avgo", "axp",  "ba",    "bac",   "bk",   "bkng", "blk",  "bmy",  "brk.b","c",
    "cat",  "cl",   "cmcsa", "cof",   "cop",  "cost", "crm",  "csco", "cvs",  "cvx",
    "de",   "dhr",  "dis",   "duk",   "emr",  "fdx",  "gd",   "ge",   "gild", "gm",
    "goog", "googl","gs",    "hd",    "hon",  "ibm",  "intu", "isrg", "iwm",  "jnj",
    "jpm",  "ko",   "lin",   "lly",   "lmt",  "low",  "ma",   "mcd",  "mdlz", "mdt",
    "met",  "meta", "mmm",   "mo",    "mrk",  "ms",   "msft", "nee",  "nflx", "nke",
    "now",  "nvda", "orcl",  "pep",   "pfe",  "pg",   "pltr", "pm",   "pypl", "qcom",
    "qqq",  "rtx",  "sbux",  "schw",  "so",   "spg",  "spy",  "t",    "tgt",  "tmo",
    "tmus", "tsla", "txn",   "uber",  "unh",  "unp",  "ups",  "usb",  "v",    "vix",
    "vz",   "wfc",  "wmt",   "xom",
]

total = len(TICKERS) * 2
completed = 0

for ticker in TICKERS:
    for kind in ("options", "underlying"):  # each ticker has two files to download
        completed += 1
        url  = f"{BASE_URL}/{ticker}/{kind}.parquet"
        dest = DATA_DIR / ticker / f"{kind}.parquet"

        # Skip if file already exists and is valid size (>1 MB or >100 KB)
        if dest.exists() \
        and ((kind == "options" and dest.stat().st_size > 1024*1024) \
        or (kind == "underlying" and dest.stat().st_size > 100*1024)):
            print(f"[{completed:>3}/{total}] skipped  {ticker}/{kind}.parquet ({dest.stat().st_size / 1e6:.1f} MB on disk)")
            continue

        print(f"[{completed:>3}/{total}] downloading {ticker}/{kind}.parquet ...")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req) as response, open(dest, "wb") as f:
                f.write(response.read())
            print(f"            done ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"            FAILED — {e}")

print("\nAll done.")