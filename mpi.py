from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

val = 1
vals = comm.gather(val, root=0)

if rank == 0:
    print(sum(vals))
