# Adapted from https://www.geeksforgeeks.org/dsa/bucket-sort-in-python/

import sys
from commons import *

def insertion_sort(bucket):
    for i in range(1, len(bucket)):
        key = bucket[i]
        j = i - 1
        while j >= 0 and bucket[j] > key:
            bucket[j + 1] = bucket[j]
            j -= 1
        bucket[j + 1] = key

def bucket_sort(arr):
    n = NUM_BUCKETS
    buckets = [[] for _ in range(n)]

    # Put array elements in different buckets
    for num in arr:
        bi = int(n * num)
        if bi >= n: bi = n - 1
        
        buckets[bi].append(num)

    # Sort individual buckets using insertion sort
    for bucket in buckets:
        insertion_sort(bucket)

    # Concatenate all buckets into arr[]
    index = 0
    
    for bucket in buckets:
        for num in bucket:
            arr[index] = num
            index += 1


NUM_BUCKETS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

code_begin = time()
arr = generate_random_values(NUM_VALUES)
bucket_sort(arr)
print(f"{1};Total(serial);{(time() - code_begin)*1e3:.3f}")

# print("Is the array sorted?", check_sorted(arr))