#!/bin/bash
#SBATCH -p mainq
#SBATCH --ntasks=1
#SBATCH -t 48:00:00 
#SBATCH -o output_log/misc.out
#SBATCH -e output_log/misc.err
#SBATCH --nodelist=raptor00

###bash ./hostgen.sh

source $HOME/.bashrc

cd /home/psadh003/tng50/dwarf_formation/

$HOME/python3/bin/python3 hello_world.py

