# Produit matrice-vecteur v = A.u
from time import time
import numpy as np

size = 1
def print_jolie(operation: str, beg: float, end: float) -> None:
    print(f"{size};{operation};{(end-beg)*1e3:.3f}")


# Dimension du problème (peut-être changé)
dim = 320

# Initialisation de la matrice (un peu plus explicite...)
A = np.zeros((dim, dim))
beg = time()

for j in range(dim):
    for i in range(dim):
        A[j, i] = (i + j) % dim + 1.


# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])


# Produit matrice-vecteur
v = A.dot(u)
end = time()

print_jolie("Total(Serial)", beg, end)
