#!/bin/bash -l
#SBATCH -J RUNewTst
#SBATCH -p saleslab
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=7
###SBATCH --mem=100gb
###SBATCH --mem-per-cpu=G
#SBATCH --time=99:00:00
#SBATCH -o output_log/inner.out
#SBATCH -e output_log/inner.err
#SBATCH --mail-user=psadh003@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --nodelist=r16

# Load needed modules
# You could also load frequently used modules from within your ~/.bashrc
module load slurm # Should already be loaded
module load openmpi # Should already be loaded
#module load hdf5

# Swtich to the working directory
cd /rhome/psadh003/bigdata/tng50/dwarf_formation/

# Run job utilizing all requested processors
# Please visit the namd site for usage details: http://www.ks.uiuc.edu/Research/namd/
# FOF=1
# python3 Ndm_extractor.py $FOF
# python3 misc.py $FOF
# python3 Pradyumna_save.py
# python3 Pradyumna_plot.py
#python3 misc.py
python3 energy_distribution_fof.py
# python3 poster_plots.py