#!/bin/bash

conda activate xesmf_env

cd /gws/nopw/j04/aopp/jessew/stuff_for_jesse/

for mylat in `seq -85.0 10.0 85.0`

do

    for mylon in `seq 5.0 10.0 355.0`
    
    do
     
        cp ./GTO_data_10deg_train/submit_GTO.sh ./GTO_data_10deg_train/slurm_scripts/submit_lat${mylat}_lon${mylon}.sh

        ## Replace lines in the submission scripts appropriately
        sed -i "11 s+.*+python /gws/nopw/j04/aopp/jessew/stuff_for_jesse/calculate_GTO_point.py --lat ${mylat} --lon ${mylon} +g" ./GTO_data_10deg_train/slurm_scripts/submit_lat${mylat}_lon${mylon}.sh

        cd ./GTO_data_10deg_train/slurm_scripts
        sbatch submit_lat${mylat}_lon${mylon}.sh
        cd /gws/nopw/j04/aopp/jessew/stuff_for_jesse/
        
    done
done