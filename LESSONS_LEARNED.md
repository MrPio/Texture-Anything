# How I was able to install `mpi4py` on Cineca's Leonardo HPC

```bash
# Load mpi compiler
module load intel-oneapi-mpi/2021.10.0
# if the module is not to be found, run
module load profile/deeplrn cineca-ai

# This gets stuck...
pip install mpi4py

# ...so I get mpi4py source...
wget https://github.com/mpi4py/mpi4py/releases/download/4.0.3/mpi4py-4.0.3.tar.gz
tar -xf mpi4py-4.0.3.tar.gz
cd mpi4py-4.0.3

# ...set MPI compiler explicitly, build and install manually
export MPICC=$(which mpicc)
python setup.py build
python setup.py install --user

# Check installation with
find ~/.local/lib -name "mpi4py"
```

This install `mpi4py` in `~/.local` and not in venv. That's ok, but to be found, check that the path is in `python -m site`, otherwise run `export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH` or write it at the bottom of `~/.bashrc` to make it permanent.

## Test script

```python
from mpi4py import MPI
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

df = pd.DataFrame({"value": range(rank * 2, rank * 2 + 2)})
dfs = comm.gather(df, root=0)

if rank == 0:
    print(pd.concat(dfs, ignore_index=True))
```

Run with `srun -n 2 python mpi.py`.
