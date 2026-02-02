# Produit matrice-vecteur v = A.u
from time import time
import numpy as np
from mpi4py import MPI

# Dimension du problème (peut-être changé)
dim = 320

# Paramètres MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def print_jolie(message: str, end:float, beg:float) -> None:
    print(f"[RANK | {rank}] {message} = {(end-beg)*1e3:.2f} ms")



N_loc = dim // size # job for each processor



# Initialisation de la matrice -- colonnes distribuées
result = np.zeros(dim)
offset_A = np.zeros((dim, N_loc))

for j in range(dim): #line
    for i in range(N_loc): #column
        offset_A[j, i] = (rank*N_loc + i + j) % dim + 1.

# print(f"[RANK | {rank}] offset_A = {offset_A}")
# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])

# Produit matrice-vecteur local
local_result = np.empty(N_loc, dtype=np.double)

beg = time()

for i in range(N_loc):
    sum = 0
    for j in range(dim):
        sum += offset_A[j, i] * u[j]
    
    local_result[i] = sum
end = time()
# print_jolie("Temp de Calcul", end, beg)

# result[0:N_loc] = result[0:N_loc] + result
# Master
if rank == 0:
    result[0:N_loc] = local_result

    # Recover data
    for src in range(1, size):
        tmp_result = np.empty(N_loc, dtype=np.double)
        comm.Recv(tmp_result, source=src)
        result[src*N_loc:(src+1)*N_loc] = tmp_result
        # print(f"[RANK | {rank}] Received from {src} : {tmp_result}")
        
    end = time()
    print_jolie("Temp Total", end, beg)
    # print(f"v = {result}")

# Servers
else:
    comm.Send(local_result, dest=0)