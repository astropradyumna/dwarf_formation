#!/bin/bash -l
#SBATCH -J RUNewTst
#SBATCH -p saleslab
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=7
#SBATCH --time=48:00:00
#SBATCH --mem=300gb
#SBATCH -o output_log/t50hiso_0.out
#SBATCH -e output_log/t50hiso_0.err
#SBATCH --mail-user=psadh003@ucr.edu
#SBATCH --mail-type=ALL

# Load needed modules
# You could also load frequently used modules from within your ~/.bashrc
module load slurm # Should already be loaded
module load openmpi # Should already be loaded
#module load hdf5

# Swtich to the working directory
cd /bigdata/saleslab/psadh003/tng50/dwarf_formation
source ~/.bashrc
python3 main_now_everything.py 0
# python3 a3_mstar_new.py 2
# python3 main_stellar_mass_new.py 2