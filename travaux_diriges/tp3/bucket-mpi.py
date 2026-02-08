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

def check_sorted(arr):
    return np.all(arr[:-1] <= arr[1:])

def pprint(*args, **kwargs):
    print(f"[{rank:02d}]", *args, **kwargs)

def zprint(*args, **kwargs):
    if rank == 0:
        pprint(*args, **kwargs)

def generate_random_values(size: int) -> list:
    data = np.random.default_rng(seed=RAND_SEED).integers(0, 100, NUM_VALUES)
    data = data.astype(np.double)
    return data




# in this moment we consider they receive all the same value


# Local/Global variables
offset_values =  NUM_VALUES // size
global_max = None
global_min = None

small_bucket = [np.array([]) for _ in range(NUM_BUCKETS)]



# 1) Generate random data
if rank == 0:
    # generate the array unsorted
    data = generate_random_values(NUM_VALUES)
    
    global_max = data.max()
    global_min = data.min()
    # print(f"extremes = ({global_min}, {global_max})", data )
    
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
#TODO: use REDUCEAll/Reduce
global_max = comm.bcast(global_max, root=0)
global_min = comm.bcast(global_min, root=0)


# print(f"Rank {rank} = {local_data}, with {global_min} | {global_max}")

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
# print(small_bucket)

# 4. Local Ordenation
small_bucket[rank].sort()

sorted_bucket = small_bucket[rank]

pprint(f"Bucket {rank} has size {len(small_bucket[rank])} and data:\n {sorted_bucket} \n")


# 4.1 Gather the size of each bucket at each rank
bucket_sizes = np.zeros(size, dtype=int)
comm.Allgather(np.array(len(sorted_bucket), dtype=int), bucket_sizes)
displacements = np.zeros(size, dtype=int)
displacements[1:] = np.cumsum(bucket_sizes[:-1])


# 5. Send back to rank = 0
if rank == 0:
    # Prepare the buffer to receive the sorted data
    sorted_data = np.empty(NUM_VALUES, dtype=np.double)   
else:
    sorted_data = None  # Initialize sorted_data for other ranks


# Gather the sorted buckets from all processors   
comm.Gatherv(
    [sorted_bucket, len(sorted_bucket), MPI.DOUBLE],  # Local sorted bucket
    [sorted_data, bucket_sizes, displacements, MPI.DOUBLE],  # Buffer to receive sorted data, sizes, displacements, and data type
    root=0
)
    
zprint("Sorted data:", sorted_data)
if rank == 0:
    print("Is the data sorted?", check_sorted(sorted_data))



