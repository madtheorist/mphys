'''
A script to calculate the GTO at a given point. Edited to only train on 80% of idx, for validation purposes.

Command line arguments:
--lat <latitude of gridpoint>
--lon <longitude of gridpoint>

Use the conda environment 'test_env' for this.

Jesse Wang

'''

import sys, importlib
from utils import point_delta_precip_vector, calculate_GTO
import matplotlib as mpl
import os
import argparse
import math
import numpy as np
import xarray as xr
from scipy import stats, optimize
import warnings

warnings.filterwarnings('ignore') # filter out deprecation warnings for now

if __name__ == "__main__":
    
    ## Parse lat/lon input from command line
    parser = argparse.ArgumentParser(
                    prog = 'calculate_GTO_point',
                    description = 'Calculates the GTO for user-specific point of the lat-lon grid')
    parser.add_argument('--lat', type=float)
    parser.add_argument('--lon', type=float)
    args = parser.parse_args()
    mylat, mylon = args.lat, args.lon
    
    ## Loading in the data
    delta_SST = xr.open_dataarray("delta_SST_pr.nc").sel(idx=slice(0, int(5545*0.8)-1)).compute() #only select 80% of data for training
    delta_precip = xr.open_dataarray("delta_precip_10deg.nc").sel(idx=slice(0, int(5545*0.8)-1)).compute()
    
    # Calculate the GTO and save it in GTO_data folder
    delta_precip_vector = point_delta_precip_vector(delta_precip, mylat=mylat, mylon=mylon, time_avg='ANNUAL', weight_by_lat=True)
    G = calculate_GTO(delta_SST, delta_precip_vector, verbose=False,
                 lat_range = [-60, 60], lon_range =[0, 360], sig_test=True, return_GTO=True)
    
    # Save as file
    lat_filename = str(mylat).replace(".","")
    lon_filename = str(mylon).replace(".","")
    
    cwd = os.getcwd()
    filepath = os.path.join(cwd, "GTO_data_10deg_train", "GTO_lat{}_lon{}.nc".format(lat_filename, lon_filename))
    G.to_netcdf(filepath)