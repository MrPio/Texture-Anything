# Lessons Learned

- [Lessons Learned](#lessons-learned)
  - [Blender has difficulties in read OBJ](#blender-has-difficulties-in-read-obj)
  - [Work with multiple site-packages (one in venv, one global)](#work-with-multiple-site-packages-one-in-venv-one-global)
  - [How I was able to achieve massive parallelism](#how-i-was-able-to-achieve-massive-parallelism)
  - [How I was able to install `mpi4py` on Cineca's Leonardo HPC](#how-i-was-able-to-install-mpi4py-on-cinecas-leonardo-hpc)
    - [Test script](#test-script)


## Blender has difficulties in read OBJ
Shapenetcore files are in the OBJ format. Trimesh has no problems in reading them, but blender's renderings are glitchy. Thus, I first exported them as GLB files using

```python
trimesh.load(obj).export(file_path, file_type="glb")
```

Then when I load one of them into blender(BPY), I merge vertices by distance to avoid black faces in CYCLES renderings.

See more in [`ShapeNetCoreDataset3D::download()`](src/dataset/shapenetcore_dataset3d.py) and in [`ShapeNetCoreObject3D::__init__()`](src/blender/object3d/shapenetcore_object3d.py).



## Work with multiple site-packages (one in venv, one global)

Do `export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH`.
Furthermore, to enable VSCode IntelliSense, go to `settings.json` and add:

```json
    "python.analysis.extraPaths": [
        "~/.local/lib/python3.11/site-packages"
    ]
```

## How I was able to achieve massive parallelism

Set `--ntasks-per-node=16` and GRES amount.

Example with 1024 parallel tasks:
`srun -n 1024 --ntasks-per-node=16 --partition boost_usr_prod  --time=00:00:15 python mpi.py`

>[!NOTE]
>`lrd_all_serial` is nearer to login node than `boost_usr_prod` is, so its bootstrap is faster.

## How I was able to install `mpi4py` on Cineca's Leonardo HPC

```bash
# Load mpi compiler (each session)
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

### Test script

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
