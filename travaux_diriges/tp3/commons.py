from time import time
import numpy as np


global NUM_VALUES, RAND_SEED
global sizem, rank, comm


NUM_BUCKETS = 4
NUM_VALUES = 8 * 1000
RAND_SEED  = 42

def generate_random_values(size: int) -> list:
    data = np.random.default_rng(seed=RAND_SEED).random(size)*1000
    data = data.astype(np.double)
    return data

def check_sorted(arr):
    return np.all(arr[:-1] <= arr[1:])
