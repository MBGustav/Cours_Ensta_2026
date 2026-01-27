from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
import matplotlib.cm
from MandelBrotSet import *
from PIL import Image

# Separation by rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
width, height = 1024, 1024
scaleX = 3./width
scaleY = 2.25/height

range_w     = width //size
offset_x    = rank * range_w          
remaining_x = width % size # in case we dont have multiple


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Parameters
width, height = 1024, 1024
scaleX = 3./width
scaleY = 2.25/height


mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)

# Local to store value
convergence = np.empty(( width if rank == 0 else range_w, height), dtype=np.double)

for y in range(height):
    for x in range(range_w):
        c = complex(-2. + scaleX*(x+offset_x), -1.125 + scaleY * y)
        convergence[x, y] = mandelbrot_set.convergence(c, smooth=True)
    


if rank == 0: # master
    #inicio - tempo
    # store data for image
    tmp_convergence = np.empty((range_w, height), dtype=np.double)
    
    #receive data from others(except master)
    for proc_i in range(1,size):
        comm.Recv(tmp_convergence, source=proc_i)
        
        convergence[range_w * (proc_i): range_w * (proc_i + 1)] = tmp_convergence
        
    # image = Image.fromarray(np.uint8(matplotlib.cm.plasma(tmp_convergence.T)*255))
    # image.show()
    

    #termino - tempo    
    #show image
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
    image.show()

else : #workers 
    send_to = 0
    # Send values to master
    comm.Send(convergence, send_to, 0)
    
    
    


                
