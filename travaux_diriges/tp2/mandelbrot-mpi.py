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
    

def print_jolie(message: str, end:float, beg:float) -> None:
    print(f"[RANK | {rank}] {message} = {(end-beg)*1e3:.2f} ms")



# Parameters
width, height = 1024, 1024
scaleX = 3./width
scaleY = 2.25/height

range_w     = width //size
offset_x    = rank * range_w          
remaining_x = width % size #TODO: in case we dont have multiple


    

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
beg = time()
for y in range(height):
    for x in range(range_w):
        c = complex(-2. + scaleX*(x+offset_x), -1.125 + scaleY * y)
        convergence[x, y] = mandelbrot_set.convergence(c, smooth=True)
end = time()
print_jolie("Temp de Calcul", end, beg)




if rank == 0: # master
 
    # store tmp data 
    tmp_convergence = np.empty((range_w, height), dtype=np.double)
    beg = time()
    #rcv data from others(except master)
    for proc_i in range(1,size):
        comm.Recv(tmp_convergence, source=proc_i)
        convergence[range_w * (proc_i): range_w * (proc_i + 1)] = tmp_convergence
        
        # image = Image.fromarray(np.uint8(matplotlib.cm.plasma(tmp_convergence.T)*255)).show()
    
    end = time()
    #show image
    print_jolie("Temp de Reception" ,end, beg)
    Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255)).show()
    
    print("Execution Ended")

else : #workers 
    send_to = 0
    # Send values to master
    beg = time()
    comm.Send(convergence, send_to, 0)
    end = time()
    print_jolie("Temp de Envoi", end, beg)
    
    
    
# EXERCICES ÉCRITES:
'''

1. (En haut)
[RANK | 7] Temp de Calcul       = 757.57 ms
[RANK | 7] Temp de Envoi        = 18.92 ms
[RANK | 2] Temp de Calcul       = 873.35 ms
[RANK | 2] Temp de Envoi        = 19.04 ms
[RANK | 3] Temp de Calcul       = 989.41 ms
[RANK | 3] Temp de Envoi        = 18.97 ms
[RANK | 5] Temp de Calcul       = 964.35 ms
[RANK | 5] Temp de Envoi        = 19.61 ms
[RANK | 4] Temp de Calcul       = 972.22 ms 
[RANK | 4] Temp de Envoi        = 19.38 ms
[RANK | 1] Temp de Calcul       = 874.09 ms
[RANK | 1] Temp de Envoi        = 1.40 ms
[RANK | 6] Temp de Calcul       = 1084.40 ms (+++)
[RANK | 6] Temp de Envoi        = 1.22 ms
[RANK | 0] Temp de Calcul       = 652.88 ms   (---)
[RANK | 0] Temp de Reception    = 530.22 ms


1.b)  Comme on peut voir, il n'y a pas un courbe pour diviser également une tache de travaux. Tout d'abbord, 
on peut voir que les travailleurs qui sont designées las taches plus lourdes s'encontrent au milieur de l'image.
Une idée  qui nous permet d'avoir une meilleur division de taches sont relationées à division dinamique de travail:
- une frontier dinamique que nous permet de diviser en temp d'execution. 
- Une autre approche de calcul, où on divise par rapport la voisinage et la condition de convergence. Cest à dire qu'on 
parametrize parties de notre image a savoir quel sont les points plus candidat a convergence.

Une bloque en analise et on construit une queue pour chaque bloque
que nous permet de classifier une meilleur classification d'endroit de convergence.
+---+---+ 
| C | D | 
+---+---+
| C | C |
+---+---+
1.c)

3. Autre archive 
'''
    
    


                
