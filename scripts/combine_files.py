from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

files = sorted(DATA_DIR.glob("PRICE_AND_DEMAND_*_VIC1.csv"))

print("Data directory:", DATA_DIR)
print("Files found:", len(files))

for file in files:
    print(file.name)

if not files:
    raise FileNotFoundError(f"No matching CSV files found in {DATA_DIR}")

df = pd.concat(
    [pd.read_csv(file) for file in files],
    ignore_index=True,
)

df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
print("Duplicate source timestamps before deduplication:", df["SETTLEMENTDATE"].duplicated().sum())

df = df.sort_values("SETTLEMENTDATE").drop_duplicates(subset="SETTLEMENTDATE").set_index("SETTLEMENTDATE")

expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="5min")

missing = expected_index.difference(df.index)

print("Expected:", len(expected_index))
print("Actual:", len(df))
print("Missing:", len(missing))

print("Duplicates:", df.index.duplicated().sum())

output_file = PROJECT_ROOT / "data" / "processed" / "PRICE_AND_DEMAND_2025_26_VIC1.csv"

output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=True)

print(f"Saved to: {output_file}")
