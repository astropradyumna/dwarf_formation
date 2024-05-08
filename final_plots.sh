#!/bin/bash -l
#SBATCH -J RUNewTst
#SBATCH -p saleslab
#SBATCH --ntasks=64
#SBATCH --mem=100gb
#SBATCH --time=48:00:00
#SBATCH -o output_log/fp.out
#SBATCH -e output_log/fp.err
#SBATCH --mail-user=psadh003@ucr.edu
#SBATCH --mail-type=ALL

# Load needed modules
# You could also load frequently used modules from within your ~/.bashrc
module load slurm # Should already be loaded
module load openmpi # Should already be loaded
#module load hdf5

# Swtich to the working directory
cd /rhome/psadh003/bigdata/tng50/dwarf_evolution/

# Run job utilizing all requested processors
# Please visit the namd site for usage details: http://www.ks.uiuc.edu/Research/namd/
# python3 main.py
python3 final_plots.py 1