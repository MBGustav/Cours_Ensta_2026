import pandas as pd
import glob

# padrão dos arquivos
file_pattern = "timing_rank_*.txt"
files = glob.glob(file_pattern)
dfs = []

for f in files:
    # ignora a primeira linha se ela for cabeçalho
    df = pd.read_csv(f, header=0)  # header=0 -> primeira linha é o nome das colunas
    dfs.append(df)

# concatena todos os arquivos
all_data = pd.concat(dfs, ignore_index=True)

# verifica colunas
print(all_data.columns)

# agrupa por iter e calcula a média
metrics = all_data.groupby("iter")[["ant_move_s","phen_sync_s","evaporation_s","total_s"]].mean()

# métricas globais
global_metrics = metrics.mean()
std_metrics = metrics.std()

print("\nMédia global:")
print(global_metrics)
print("\nDesvio padrão:")
print(std_metrics)

# salva CSV
metrics.to_csv("mpi_average_per_iteration.csv")
global_metrics.to_csv("mpi_global_average.csv")
