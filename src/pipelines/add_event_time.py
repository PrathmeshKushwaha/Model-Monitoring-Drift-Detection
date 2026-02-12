import pandas as pd

INPUT_PATH = "data/reference.csv"
OUTPUT_PATH = "data/reference_with_time.csv"

df = pd.read_csv(INPUT_PATH)

df["event_time"] = pd.date_range(
    start="2024-01-01",
    periods=len(df),
    freq="H"
)

df.to_csv(OUTPUT_PATH, index=False)

print("event_time added and saved to", OUTPUT_PATH)
