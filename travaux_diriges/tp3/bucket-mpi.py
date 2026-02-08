from mpi4py import MPI
from time import time
import numpy as np




# Parallel MPI

# 1. Generate the random values at r=0
# 2. Distribute parts of array based on processor quantity
# 3. separate the buckets based on global_min/global_max and buckets quantity (?)
# 4. local ordenation
# 5. send back to r=0
# 6. its done :)

# Paramètres MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Sorting Parameters
NUM_BUCKETS = size
NUM_VALUES = size*1000
RAND_SEED  = 42
DATA_TYPE = np.double

# Local/Global variables
offset_values =  NUM_VALUES // size

small_bucket = [np.array([]) for _ in range(NUM_BUCKETS)]

def check_sorted(arr):
    return np.all(arr[:-1] <= arr[1:])

def pprint(*args, **kwargs):
    print(f"[{rank:02d}]", *args, **kwargs)

def zprint(*args, **kwargs):
    if rank == 0:
        pprint(*args, **kwargs)

def generate_random_values(size: int) -> list:
    data = np.random.default_rng(seed=RAND_SEED).random(size)*1000
    data = data.astype(np.double)
    return data


# 1) Generate random data
if rank == 0:
    # generate the array unsorted
    data = generate_random_values(NUM_VALUES)
else:
    data = None


# 2) Send data to the others processors
if rank == 0:
    
    for p in range(1, size):
        i_beg = p * offset_values
        i_end = (p+1) * offset_values
        comm.Send(data[i_beg:i_end], p, 0)
    
    local_data = data[0:offset_values]
    # data = data[(rank) * offset:(rank+1) * offset]
else:
    local_data = np.empty(offset_values, dtype=np.double)
    comm.Recv(local_data, source=0)


# 3) give back the values to the expected buckets
# 3.1 find global max and min
local_max = local_data.max()
local_min = local_data.min()
global_max = comm.allreduce(local_max, op=MPI.MAX)
global_min = comm.allreduce(local_min, op=MPI.MIN)

# 3.2 separate the buckets based on global_min/global_max 
for k in local_data:
    # Calculate idx for separation
    norm = (k - global_min)/(global_max - global_min) 
    
    idx = int(norm * NUM_BUCKETS)
    if idx >=  NUM_BUCKETS:
        idx = NUM_BUCKETS -1
    small_bucket[idx] = np.append(small_bucket[idx], k)

# Each processor sends its bucket[i] to processor i
for i in range(NUM_BUCKETS):
    if i == rank:
        # Receive from all other processors
        for p in range(size):
            if p != rank:
                received = comm.recv(source=p)
                small_bucket[i] = np.append(small_bucket[i], received)
    else:
        # Send this bucket to processor i
        comm.send(small_bucket[i], dest=i)

# Synchronize all processes
comm.Barrier()


# 4. Local Ordenation
sorted_bucket = np.sort(small_bucket[rank])

# 4.1 Gather the size of each bucket at each rank
bucket_sizes = np.zeros(size, dtype=int) # local array for each bucket
comm.Allgather(np.array(len(sorted_bucket), dtype=int), bucket_sizes) # Gather the size of each bucket at each rank
displacements = np.zeros(size, dtype=int) # calc displacements for Gatherv
displacements[1:] = np.cumsum(bucket_sizes[:-1]) # offset for the 1st element


# 5. Send back to rank = 0
if rank == 0:
    sorted_data = np.empty(NUM_VALUES, dtype=np.double)
else:
    sorted_data = None # Initialize sorted_data for other ranks


#6. Gather the sorted buckets from all processors   
comm.Gatherv(
    [sorted_bucket, len(sorted_bucket), MPI.DOUBLE],  # Local sorted bucket
    [sorted_data, bucket_sizes, displacements, MPI.DOUBLE],  # Buffer to receive sorted data, sizes, displacements, and data type
    root=0
)
    
zprint("Sorted data:", sorted_data)
if rank == 0:
    print("Is the data sorted?", check_sorted(sorted_data))