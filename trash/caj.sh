#!/bin/bash -l
#SBATCH -J RUNewTst
#SBATCH -p saleslab
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
###SBATCH --mem=100gb
###SBATCH --mem-per-cpu=G
#SBATCH --time=48:00:00
#SBATCH -o output_log/caj.out
#SBATCH -e output_log/caj.err
#SBATCH --mail-user=psadh003@ucr.edu
#SBATCH --mail-type=ALL

# Load needed modules
# You could also load frequently used modules from within your ~/.bashrc
module load slurm # Should already be loaded
module load openmpi # Should already be loaded
#module load hdf5

# Swtich to the working directory
cd /rhome/psadh003/bigdata/tng50/dwarf_formation/

# Run job utilizing all requested processors
# Please visit the namd site for usage details: http://www.ks.uiuc.edu/Research/namd/
python3 checking_against_jess.py