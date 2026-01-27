# Produit matrice-vecteur v = A.u
import numpy as np

# Dimension du problème (peut-être changé)
dim = 120
# Initialisation de la matrice (un peu plus explicite...)
A = np.zeros((dim, dim))
for j in range(dim):
    for i in range(dim):
        A[j, i] = (i + j) % dim + 1.
print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
print(f"u = {u}")

# Produit matrice-vecteur
v = A.dot(u)
print(f"v = {v}")
