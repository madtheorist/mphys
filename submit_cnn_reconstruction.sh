#!/bin/bash 
#SBATCH --partition=short-serial
#SBATCH --job-name=reconstruction
#SBATCH --time=4:00:00 
#SBATCH --mem-per-cpu=10000

conda activate xesmf_env
cd /gws/nopw/j04/aopp/jessew/stuff_for_jesse/
python /gws/nopw/j04/aopp/jessew/stuff_for_jesse/reconstruction_cnn.py --lat 15 --lon 295