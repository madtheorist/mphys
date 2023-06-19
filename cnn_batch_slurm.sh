#!/bin/bash

conda activate xesmf_env

for mylat in `seq -85.0 10.0 85.0`

do

    for mylon in `seq 5.0 10.0 355.0`
    
    do
     
        cp submit_cnn_reconstruction.sh CNN_slurm_validation/submit_lat${mylat}_lon${mylon}.sh

        ## Replace lines in the submission scripts appropriately
        sed -i "9 s+.*+python /gws/nopw/j04/aopp/jessew/stuff_for_jesse/reconstruction_cnn.py --lat ${mylat} --lon ${mylon} +g" ./CNN_slurm_validation/submit_lat${mylat}_lon${mylon}.sh

        cd CNN_slurm_validation
        sbatch submit_lat${mylat}_lon${mylon}.sh
        cd ..
        
    done
done