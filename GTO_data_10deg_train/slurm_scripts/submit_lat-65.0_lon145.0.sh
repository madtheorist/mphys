#!/bin/bash 
#SBATCH --partition=short-serial
#SBATCH --job-name=piecewiseGTO
#SBATCH --time=1:00:00 
#SBATCH --mem-per-cpu=8000

conda activate test_env

cd /gws/nopw/j04/aopp/jessew/stuff_for_jesse/

python /gws/nopw/j04/aopp/jessew/stuff_for_jesse/calculate_GTO_point.py --lat -65.0 --lon 145.0 