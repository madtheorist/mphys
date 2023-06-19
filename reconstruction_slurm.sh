#!/bin/bash 
#SBATCH --partition=short-serial
#SBATCH --job-name=reconstruction
#SBATCH -o rvalues.out 
#SBATCH -e rvalues.err
#SBATCH --time=24:00:00 
#SBATCH --mem-per-cpu=10000

# Activate your conda environment
conda activate xesmf_env

for mylat in `seq -85.0 10.0 85.0`
do
    for mylon in `seq 5.0 10.0 355.0`
    do 
        python /gws/nopw/j04/aopp/jessew/stuff_for_jesse/reconstruction.py --lat $mylat --lon $mylon
    done
done