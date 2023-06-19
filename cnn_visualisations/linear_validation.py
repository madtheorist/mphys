"""
Quick script to compute r-values obtained from comparing delta_precip from GCM simulations to delta_precip as predicted by sum(linear GTO * delta_SST). 

Use e.g. test_env for this.
"""

import os
import numpy as np
import xarray as xr
from scipy import stats

if __name__ == '__main__':
    
    validation = False
    reconstruct = True
    
    os.chdir("..") # go back to parent directory
    
    # load in simulated data
    delta_SST = xr.open_dataarray("delta_SST_pr.nc").compute()
    delta_precip = xr.open_dataarray("delta_precip_10deg.nc").compute()
    
    num_examples = len(delta_SST.idx.values)
    print(num_examples)
    
    # loop over the 10x10deg delta_precip grid
    for precip_lat in delta_precip.lat.values:
        
        print("Executing Latitude {}".format(precip_lat))
        
        for precip_lon in delta_precip.lon.values:
            
            # extract GTO for delta_precip gridpoint - computed on all data
            lat_filename = str(format(precip_lat, '.1f')).replace(".","")
            lon_filename = str(format(precip_lon, '.1f')).replace(".","")
            gto_filepath = "GTO_data_10deg_train/GTO_lat{}_lon{}.nc".format(lat_filename, lon_filename)
            G = xr.open_dataarray(gto_filepath)
            
            if validation:
                # predictions on just the test set (~1000 data points)
                test_array = delta_SST.sel(idx = slice(int(0.8*num_examples), None))
                weights = np.cos(np.deg2rad(G.lat))* 6371**2 * 1 * 1 * (np.pi/180)**2 * G
                product = test_array.weighted(weights.fillna(0))
                prediction = product.sum(('lon', 'lat'), skipna=True)
                label = delta_precip.sel(time_avg='ANNUAL', 
                                         lat=precip_lat, lon=precip_lon, 
                                         idx = slice(int(0.8*num_examples), None))

                # compute r-value and write to file
                alpha, beta, r, p, se = stats.linregress(prediction, label)
                with open('CNN_models/validation_r_values_linear.txt', 'a') as f: 
                    f.write('{} {} {} {}\n'.format(precip_lon, precip_lat, r, p))
                    
            if reconstruct:
                observed_delta_SST = xr.open_dataarray('CNN_models/observed_delta_SST_1980-99.nc') #from -60 to 60
                #observed_delta_SST = xr.open_dataarray("CNN_models/hadsst_annual_SSTgrid.nc").sel(lat=slice(60, -60)) 
                #G = G.sel(lat=observed_delta_SST.lat)
                
                weights = np.cos(np.deg2rad(G.lat))* 6371**2 * 1 * 1 * (np.pi/180)**2 * G
                product = observed_delta_SST.weighted(weights.fillna(0))
                prediction = product.sum(('lon', 'lat'), skipna=True)
                
                observed_precip = xr.open_dataarray("observed_mean_annual.nc").compute()
                P_tot = observed_precip.sel(lat=precip_lat, lon=precip_lon).values
                
                try:
                    alpha, beta, r, p, se = stats.linregress(prediction.sel(time=slice('1900', '2010')), P_tot)
                except ValueError:
                    r, p = 0, 1
                    print('Regression failed for lat {} lon {}'.format(precip_lat, precip_lon))
                    
                with open('CNN_models/reconstructions_linear_delta.txt', 'a') as f: 
                    f.write('{} {} {} {}\n'.format(precip_lon, precip_lat, r, p))