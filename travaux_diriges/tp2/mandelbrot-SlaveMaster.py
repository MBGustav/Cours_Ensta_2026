from mpi4py import MPI
import numpy as np
from PIL import Image
import matplotlib.cm
from time import time
from MandelBrotSet import MandelbrotSet

# MPI init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def print_jolie(operation: str, beg: float, end: float) -> None:
    print(f"{size};{operation};{(end-beg)*1e3:.3f}")

# Mandelbrot parameters
width, height = 1024, 1024
scaleX = 3.0 / width
scaleY = 2.25 / height

# Master-Worker parameters
CHUNK_SIZE = 16  # Lines per job



mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)

def compute_lines(y_start, y_end):
    """Computa um bloco de linhas do Mandelbrot"""
    local_data = np.empty((width, y_end - y_start), dtype=np.double)
    for y in range(y_start, y_end):
        for x in range(width):
            c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y)
            local_data[x, y - y_start] = mandelbrot_set.convergence(c, smooth=True)
    return local_data


if rank == 0:
    convergence = np.empty((width, height), dtype=np.double)
    
    # Create queue of tasks
    tasks = []
    for i in range(0, height, CHUNK_SIZE): # [0, CHUNK_SIZE, 2*CHUNK_SIZE, ...]
        y_start = i
        y_end = min(i + CHUNK_SIZE, height)
        tasks.append((y_start, y_end)) # [START, END]
        
    task_idx = 0
    finished_workers = 0

    beg = time()
    
    # Send Tasks for workers
    for worker in range(1, size):
        if task_idx < len(tasks):
            comm.send(tasks[task_idx], dest=worker)
            task_idx += 1
    
    while finished_workers < size - 1:
        # Recebe resultado de qualquer worker
        status = MPI.Status()
        
        y_start, y_end, data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        src = status.Get_source()
        
        convergence[:, y_start:y_end] = data
        
        # Send new job or stop
        if task_idx < len(tasks):
            comm.send(tasks[task_idx], dest=src)
            task_idx += 1
        else:
            comm.send(None, dest=src)  # Stop sign
            finished_workers += 1
    
    
    # Mostra imagem final
    # Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T) * 255)).show()

# ================= WORKERS =================
else:
    while True:
        task = comm.recv(source=0)
        if task is None:  # Stop sign
            break
        y_start, y_end = task
        beg = time()
        local_result = compute_lines(y_start, y_end)
        end = time()
        # print_jolie("Calcul", beg, end)
        comm.send((y_start, y_end, local_result), dest=0)



if rank == 0:
    end = time()
    print_jolie("Total", beg, end)
    