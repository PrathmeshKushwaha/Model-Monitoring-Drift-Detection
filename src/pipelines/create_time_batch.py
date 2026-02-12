import os
import pandas as pd
from datetime import datetime, timedelta

INPUT_CSV = "data/reference_with_time.csv"   # path to full dataset
OUTPUT_DIR = "data/batches"             # where batch files go
BATCH_SIZE = 3000                        # IMPORTANT: avoid tiny batches
START_DATE = "2023-01-01"                # fake start date for time simulation
TIME_GAP_DAYS = 1                        # gap between batches

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_CSV)

if len(df) < BATCH_SIZE:
    raise ValueError("Dataset too small for chosen batch size")

# if "event_time" not in df.columns:
#     start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
#     df["event_time"] = [
#         start_dt + timedelta(minutes=i)
#         for i in range(len(df))
#     ]

df["event_time"] = pd.to_datetime(df["event_time"])

# Sort by time (important for monitoring)
df = df.sort_values("event_time").reset_index(drop=True)

batch_id = 0
batch_start_time = df.loc[0, "event_time"]

for i in range(0, len(df), BATCH_SIZE):
    batch_df = df.iloc[i:i + BATCH_SIZE].copy()

    # Shift timestamps per batch to simulate time progression
    batch_df["event_time"] = batch_df["event_time"] + timedelta(
        days=batch_id * TIME_GAP_DAYS
    )

    batch_filename = f"batch_{batch_id:03d}.csv"
    batch_path = os.path.join(OUTPUT_DIR, batch_filename)

    batch_df.to_csv(batch_path, index=False)

    print(f"Created {batch_filename} with {len(batch_df)} rows")

    batch_id += 1

print("\nBatch creation complete.")
print(f"Total batches created: {batch_id}")
