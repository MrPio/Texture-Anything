from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
files = list(SCRIPT_DIR.glob("*.parquet"))
pd.concat(map(pd.read_parquet, files), ignore_index=False)
df = pd.concat(map(pd.read_parquet, files), ignore_index=False)
df.index = df.index.astype(str)
df.to_parquet(SCRIPT_DIR / "statistics.parquet", index=True)
for f in files:
    if 'statistics.parquet' not in str(f):
        f.unlink()