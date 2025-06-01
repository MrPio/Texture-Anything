#!/bin/bash

#SBATCH --job-name=download_objaverse
#SBATCH --time=04:00:00
#SBATCH --mem=6G

start=$1
end=$2

logfile="download_objaverse_${start}_${end}.log"
exec > "$logfile" 2>&1

cd $SCRATCH/objaverse/
module load git-lfs/

for i in $(seq -w $start $end); do
    git lfs pull --include="glbs/000-$i/*"
done