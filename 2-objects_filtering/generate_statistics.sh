#!/bin/bash
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --partition=boost_usr_prod
##SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-19

srun python 2-objects_filtering/compute_statistics.py