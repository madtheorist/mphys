'''
A script to calculate the exponential coefficients and 'effective' GTO at a given point. Edited to only train on 80% of idx, for validation purposes.

Command line arguments:
--lat <latitude of gridpoint>
--lon <longitude of gridpoint>

Use the conda environment 'test_env' for this.

Jesse Wang

'''

import os
import sys, importlib
from utils import point_delta_precip_vector
import matplotlib as mpl
import os
import argparse
import math
import numpy as np
import xarray as xr
from scipy import stats
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore') # filter out deprecation warnings for now

def calculate_exponential(delta_SST, delta_precip_vector, 
                          lat_range = [-60, 60], lon_range =[0, 360], verbose=True):
    
    '''
    Fit exponential of form delta_precip = a * exp (b * delta_SST) + c for each delta_SST point.
    Stores the result, and the effective GTO given by the correlation coefficient of the fit, in an xarray dataset.
    Returns the xarray dataset.
    '''
    
    # Defining global variables
    T_max = 2 # maximum temperature perturbation
    a = 6371 # earth radius in kilometers
    phi_0 = math.radians(1.25) # latitude spacing in radians
    lambda_0 = math.radians(1.875) # longitude spacing in radians
    lats = delta_SST.lat.values
    lons = delta_SST.lon.values
    n_ensemble = len(delta_precip_vector)
    
    # functional form to fit
    def exponential_func(x, a, b, c):
        return a * np.exp(b*x) + c
    
    # initializing data arrays
    A = xr.DataArray(np.zeros((len(lats),len(lons))), coords=[lats, lons], dims=["lat", "lon"]) 
    B = xr.DataArray(np.zeros((len(lats),len(lons))), coords=[lats, lons], dims=["lat", "lon"]) 
    C = xr.DataArray(np.zeros((len(lats),len(lons))), coords=[lats, lons], dims=["lat", "lon"]) 
    G = xr.DataArray(np.zeros((len(lats),len(lons))), coords=[lats, lons], dims=["lat", "lon"]) 
    
    for lat in lats:
        
        if lat < lat_range[0] or lat > lat_range[1]:
            continue
            
        if verbose and (lat % 10) == 0:
            print('Starting latitude {}'.format(lat))
            
        L_x = math.cos(math.radians(lat))
        
        for lon in lons:
            
            if lon < lon_range[0] or lon > lon_range[1]:
                continue
            
            x = delta_SST.sel(lat=lat, lon=lon).values # the x values are the delta_SST values
            
            # fit functional form to data
            try:
                popt, pcov = curve_fit(exponential_func, x, delta_precip_vector, p0 = [0.5, 1, -1], maxfev=5000)
                a, b, c = popt[0], popt[1], popt[2]

                A.loc[lat, lon] = a
                B.loc[lat, lon] = b
                C.loc[lat, lon] = c
            
                delta_precip_predictions = exponential_func(x, a, b, c)

                # effective GTO weighted by latitude is correlation between predictions and actual
                g = np.cov(delta_precip_predictions, delta_precip_vector)[0,1] / L_x

                # Check statistical significance of g
                # r is cov(delta_SST, delta_precip) divided by the product of the variances

                r = np.cov(x, delta_precip_vector)[0, 1] / np.sqrt(np.cov(x, delta_precip_vector)[1,1]*np.cov(x, delta_precip_vector)[0, 0])

                t = r*np.sqrt((n_ensemble-2)/(1-r**2))
                p = 1 - stats.norm.cdf(np.abs(t))
                is_significant = np.greater_equal(5, p*100*2)

                # If statistically significant, append to G matrix
                if is_significant:
                    G.loc[lat, lon] = g
                else:
                    G.loc[lat, lon] = np.nan
            
            except RuntimeError: # if optimal parameters cannot be found, set everything to NaN
                
                A.loc[lat, lon] = np.nan
                B.loc[lat, lon] = np.nan
                C.loc[lat, lon] = np.nan
                G.loc[lat, lon] = np.nan
            
    return A, B, C, G

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
    
    A, B, C, G = calculate_exponential(delta_SST, delta_precip_vector, verbose=True,
                 lat_range = [-60, 60], lon_range =[0, 360])
    
    exponential_params = xr.concat([A, B, C, G], 'abcg_dim') #concatenate along new dimension, store in dataset
    
    # Save as files
    lat_filename = str(mylat).replace(".","")
    lon_filename = str(mylon).replace(".","")
    
    cwd = os.getcwd()
    filepath = os.path.join(cwd, "exponential_model", "parameters", "exponential_params_lat{}_lon{}.nc".format(lat_filename, lon_filename))
    exponential_params.to_netcdf(filepath)