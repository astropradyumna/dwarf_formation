#!/bin/bash -l
#SBATCH -J RUNewTst
#SBATCH -p saleslab
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=7
###SBATCH --mem=100gb
###SBATCH --mem-per-cpu=G
#SBATCH --time=48:00:00
#SBATCH -o output_log/aipvfs.out
#SBATCH -e output_log/aipvfs.err
#SBATCH --mail-user=psadh003@ucr.edu
#SBATCH --mail-type=ALL

# Load needed modules
# You could also load frequently used modules from within your ~/.bashrc
module load slurm # Should already be loaded
module load openmpi # Should already be loaded
#module load hdf5

# Swtich to the working directory
cd /bigdata/saleslab/psadh003/tng50/dwarf_formation/add_properties/

python3 add_inf_pos_vel_for_surv.py 2