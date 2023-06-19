#!/bin/bash 
#SBATCH --partition=short-serial
#SBATCH --job-name=mySlurmjob 
#SBATCH -o myjob.out 
#SBATCH -e myjob.err
#SBATCH --time=24:00:00 
#SBATCH --mem-per-cpu=8000

# Activate your conda environment
conda activate test_env

for mylat in `seq -85 10 -65`
do
    for mylon in `seq 5 10 355`
    do 
        python /gws/nopw/j04/aopp/jessew/stuff_for_jesse/calculate_GTO_point.py --lat $mylat --lon $mylon
    done
done

for mylat in `seq 65 10 85`
do
    for mylon in `seq 5 10 355`
    do 
        python /gws/nopw/j04/aopp/jessew/stuff_for_jesse/calculate_GTO_point.py --lat $mylat --lon $mylon
    done
done