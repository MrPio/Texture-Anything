from mpi4py import MPI
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

val = 1
vals = comm.gather(val, root=0)

if rank == 0:
    print(sum(vals))
