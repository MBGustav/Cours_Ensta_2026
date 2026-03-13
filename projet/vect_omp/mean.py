import pandas as pd

df = pd.read_csv("loop_time.csv")
# convert to number
df["elapsed_seconds"] = pd.to_numeric(df["elapsed_seconds"], errors='coerce')

mean_value= df["elapsed_seconds"].mean()   
max_value = df["elapsed_seconds"].max()
min_value = df["elapsed_seconds"].min()

print("Mean(ms):", mean_value * 1000)
print("Max(ms):", max_value * 1000)
print("Min(ms):", min_value * 1000)
