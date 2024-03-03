#!/bin/bash
#SBATCH -p mainq
#SBATCH --ntasks=1
#SBATCH -t 48:00:00 
#SBATCH -o output_log/misc.out
#SBATCH -e output_log/misc.err
#SBATCH --nodelist=raptor00
#SBATCH --mem=24G

###bash ./hostgen.sh

cd /home/psadh003/tng50/dwarf_formation/

python3 main.py