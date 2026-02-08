from mpi4py import MPI
from commons import * 





def print_jolie(message: str, end:float, beg:float) -> None:
    if rank == 0:
        print(f"{size};{message};{(end-beg)*1e3:.3f}")

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

# Local/Global variables

num_subbuckets = 10 * size  # sub-buckets finos para histograma
local_hist = np.zeros(num_subbuckets, dtype=int)
local_bucket = [np.array([]) for _ in range(NUM_BUCKETS)]
local_cumulative = 0

local_target_per_bucket = NUM_VALUES // size
offset_values =  NUM_VALUES // size
global_hist = np.zeros(num_subbuckets, dtype=int)

def pprint(*args, **kwargs):
    print(f"[{rank:02d}]", *args, **kwargs)

def zprint(*args, **kwargs):
    if rank == 0:
        pprint(*args, **kwargs)

# 1) Generate random data
if rank == 0:
    # generate the array unsorted
    data = generate_random_values(NUM_VALUES)
else:
    data = None


code_begin = time()
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


print_jolie("bucket-transfer" , time(), code_begin)

# 3) give back the values to the expected buckets
# 3.1 find global max and min
local_max = local_data.max()
local_min = local_data.min()
global_max = comm.allreduce(local_max, op=MPI.MAX)
global_min = comm.allreduce(local_min, op=MPI.MIN)


# IMPLEMENTATION - HISTOGRAM FOR LOAD BALANCING
# We separe new small buckets to balance the qtd. for each bucket

comm.Allreduce(local_hist, global_hist, op=MPI.SUM)
local_bucket_limits = np.linspace(global_min, global_max, NUM_BUCKETS + 1)


for i, count in enumerate(global_hist):    
    local_cumulative += count
    if local_cumulative >= local_target_per_bucket:
        
        limit = global_min + (i + 1)/num_subbuckets * (global_max - global_min)
        local_bucket_limits.append(limit)
        local_cumulative = 0
local_bucket_limits[-1] = global_max



for k in local_data:
    idx = np.searchsorted(local_bucket_limits, k, side='right') - 1
    if idx == NUM_BUCKETS:  # caso k == global_max
        idx = NUM_BUCKETS - 1
    local_hist[idx] += 1



aux_begin = time()
# 3.2 separate the buckets based on Local Histogram and the limits calculated

# For each value
for k in local_data:
    # For each sub-bucket 
    for idx in range(size):
        
        # check if the value belongs to the bucket
        if local_bucket_limits[idx] <= k < local_bucket_limits[idx+1]:
            local_bucket[idx] = np.append(local_bucket[idx], k)
            break
        
        # Handle edge case for the maximum value
        elif idx == size - 1 and k == global_max:
            local_bucket[idx] = np.append(local_bucket[idx], k)




# Each processor sends its bucket[i] to processor i
for i in range(NUM_BUCKETS):
    if i == rank:
        # Receive from all other processors
        for p in range(size):
            if p != rank:
                received = comm.recv(source=p)
                local_bucket[i] = np.append(local_bucket[i], received)
    else:
        # Send this bucket to processor i
        comm.send(local_bucket[i], dest=i)

# Synchronize all processes
comm.Barrier()
print_jolie("bucket-transfer" , time(), aux_begin)


# 4. Local Ordenation
aux_begin = time()
sorted_bucket = np.sort(local_bucket[rank])
print_jolie("local-sort" , time(), aux_begin)


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

print_jolie("Total(histogram)" , time(), code_begin)
    
# zprint("Sorted data:", sorted_data)
# if rank == 0:
    # print("Is the data sorted?", check_sorted(sorted_data))