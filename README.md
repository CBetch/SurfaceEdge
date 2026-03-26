# SurfaceEdge

## Data

This project uses historical options chain data for 104 US equities and ETFs (2008–2025), sourced from [philippdubach/options-data](https://github.com/philippdubach/options-data).

Data is not included in the repo. Run the download script once before anything else:

```bash
python download_data.py
```

This will populate `data/<ticker>/options.parquet` and `data/<ticker>/underlying.parquet` for all 104 tickers. Already-downloaded files are skipped automatically, so re-running is safe.
