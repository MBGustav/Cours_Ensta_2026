import pandas as pd

df = pd.read_csv("result_time.csv")
# convert to number
df["elapsed_seconds"] = pd.to_numeric(df["elapsed_seconds"], errors="coerce")
df = df.dropna(subset=["elapsed_seconds"]).sort_values("elapsed_seconds")
if len(df) >= 7:
    df = df.iloc[3:-3]

mean_value= df["elapsed_seconds"].mean()   
max_value = df["elapsed_seconds"].max()
min_value = df["elapsed_seconds"].min()

print("Mean(ms):", mean_value * 1000)
print("Max(ms):", max_value * 1000)
print("Min(ms):", min_value * 1000)
