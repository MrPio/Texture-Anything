from pathlib import Path
import pandas as pd
import os

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT = SCRIPT_DIR / "statistics.parquet"

if os.path.exists(OUTPUT):
    os.remove(OUTPUT)
    print("Removed old statistics.parquet")
files = list(SCRIPT_DIR.glob("*.parquet"))

df = pd.concat(map(pd.read_parquet, files), ignore_index=False)
df.index = df.index.astype(str)
df.to_parquet(OUTPUT, index=True)
print(f"Loaded {len(df)} statistics")

for f in files:
    f.unlink()
print(f"Removed {len(files)} partial statistics")
