from time import time
import numpy as np
from mpi4py import MPI

# Dimension du problème
dim = 16

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def print_jolie(message: str, end:float, beg:float) -> None:
    print(f"{size};{message};{(end-beg)*1e3:.3f}")
    # print(f"[RANK | {rank}] {message} = {(end-beg)*1e3:.2f} ms")


# Division
N_loc = dim // size

# Initialisation du vecteur u
u = np.array([i + 1. for i in range(dim)])

# Local build Mtx A
A_local = np.zeros((dim, N_loc))

for i in range(dim):          # lignes
    for j in range(N_loc):    # colonnes locales
        global_j = rank * N_loc + j
        A_local[i, j] = (i + global_j) % dim + 1.

v_local = np.zeros(dim)

beg = time()
for i in range(dim):
    for j in range(N_loc):
        v_local[i] += A_local[i, j] * u[rank * N_loc + j]
# end = time()

# print_jolie("Temps calcul local", end, beg)


# Réduction (somme vers le maître)
if rank == 0:
    v = np.zeros(dim)
else:
    v = None

comm.Reduce(v_local, v, op=MPI.SUM, root=0)

# Résultat final (uniquement maître)
if rank == 0:
    end = time()
    print_jolie("Total", end, beg)