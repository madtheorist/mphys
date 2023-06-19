"""
Quick script to compute r-values obtained from comparing delta_precip from GCM simulations to delta_precip as predicted by sum(linear GTO * delta_SST). 

Use e.g. test_env for this.
"""

import os
import numpy as np
import xarray as xr
from scipy import stats
from scipy.optimize import curve_fit

if __name__ == '__main__':
    
    validation = False
    multilinear = True
    reconstruct = True
    
    os.chdir("..") # go back to parent directory
    
    # load in simulated data
    delta_SST = xr.open_dataarray("delta_SST_pr.nc").compute()
    delta_precip = xr.open_dataarray("delta_precip_10deg.nc").compute()
    
    num_examples = len(delta_SST.idx.values)
    print(num_examples)
    
    # Define function for multilinear regression
    def multilinear_func(prediction_vec, alpha_0, alpha_1, beta):
        return alpha_0 * prediction_vec[0] + alpha_1 * prediction_vec[1] + beta
    
    # loop over the 10x10deg delta_precip grid
    for precip_lat in delta_precip.lat.values:
        
        print("Executing Latitude {}".format(precip_lat))
        
        for precip_lon in delta_precip.lon.values:
            
            # extract positive and negative GTO for delta_precip gridpoint - computed on training data (~4000 idx)
            lat_filename = str(format(precip_lat, '.1f')).replace(".","")
            lon_filename = str(format(precip_lon, '.1f')).replace(".","")
            G_pos_filepath = "GTO_piecewise_10deg/GTO_pos_lat{}_lon{}.nc".format(lat_filename, lon_filename)
            G_neg_filepath = "GTO_piecewise_10deg/GTO_neg_lat{}_lon{}.nc".format(lat_filename, lon_filename)
            G_pos = xr.open_dataarray(G_pos_filepath)
            G_neg = xr.open_dataarray(G_neg_filepath)
            
            
            if validation:
                
                # predictions on just the test set (~1000 idx)
                test_array = delta_SST.sel(idx = slice(int(0.8*num_examples), None))

                delta_SST_pos = test_array.where(test_array > 0, other=0) #delta_SST array with the only non-zero values being positive
                delta_SST_neg = test_array.where(test_array < 0, other=0)

                weights = np.cos(np.deg2rad(G_pos.lat))* 6371**2 * 1 * 1 * (np.pi/180)**2 * G_pos
                product = delta_SST_pos.weighted(weights.fillna(0))
                prediction_pos = product.sum(('lon', 'lat'), skipna=True)

                weights = np.cos(np.deg2rad(G_neg.lat))* 6371**2 * 1 * 1 * (np.pi/180)**2 * G_neg
                product = delta_SST_neg.weighted(weights.fillna(0))
                prediction_neg = product.sum(('lon', 'lat'), skipna=True)

                prediction = prediction_pos + prediction_neg # final prediction

                label = delta_precip.sel(time_avg='ANNUAL', 
                                         lat=precip_lat, lon=precip_lon, 
                                         idx = slice(int(0.8*num_examples), None))

                # compute r-value and write to file
                alpha, beta, r, p, se = stats.linregress(prediction, label)
                with open('CNN_models/validation_r_values_piecewise.txt', 'a') as f: 
                    f.write('{} {} {} {}\n'.format(precip_lon, precip_lat, r, p))
                    
            if reconstruct:
                
                observed_delta_SST = xr.open_dataarray('CNN_models/observed_delta_SST_1980-99.nc') #from -60 to 60
                
                # isolate positive and negative SST regions
                delta_SST_pos = observed_delta_SST.where(observed_delta_SST > 0, other=0) 
                delta_SST_neg = observed_delta_SST.where(observed_delta_SST < 0, other=0)
                
                # make lat grids the same
                G_pos = G_pos.sel(lat=observed_delta_SST.lat)
                G_neg = G_neg.sel(lat=observed_delta_SST.lat)
                
                weights = np.cos(np.deg2rad(G_pos.lat))* 6371**2 * 1 * 1 * (np.pi/180)**2 * G_pos
                product = delta_SST_pos.weighted(weights.fillna(0))
                prediction_pos = product.sum(('lon', 'lat'), skipna=True) # dimension (149, )

                weights = np.cos(np.deg2rad(G_neg.lat))* 6371**2 * 1 * 1 * (np.pi/180)**2 * G_neg
                product = delta_SST_neg.weighted(weights.fillna(0))
                prediction_neg = product.sum(('lon', 'lat'), skipna=True) # dimension (149, )
                
                
                # Single variable regression
           
                observed_precip = xr.open_dataarray("observed_mean_annual.nc").compute()
                P_tot = observed_precip.sel(lat=precip_lat, lon=precip_lon).values
                
                
                if multilinear:
                    
                    prediction_pos = prediction_pos.sel(time=slice('1900', '2010')).values.reshape(1, -1)
                    prediction_neg = prediction_neg.sel(time=slice('1900', '2010')).values.reshape(1, -1)         
                    prediction_vec = np.concatenate((prediction_pos, prediction_neg), axis=0)

                    popt, pcov = curve_fit(multilinear_func, prediction_vec, P_tot)
                    
                    alpha_1, alpha_2, beta = popt[0], popt[1], popt[2]
                    
                    P_recon = alpha_1 * prediction_pos + alpha_2 * prediction_neg + beta
                    
                    with np.errstate(invalid='ignore'): #ignore runtime warning
                        _, _, r, p, _ = stats.linregress(P_recon, P_tot)
                    
                    with open('CNN_models/reconstructions_piecewise_multilinear.txt', 'a') as f: 
                        f.write('{} {} {} {}\n'.format(precip_lon, precip_lat, r, p))
                    
                
                # define
                
                else: # single variable regression
                    prediction = prediction_pos + prediction_neg # prediction for years 1870-2018 (149 time dim)
                    
                    try:
                        alpha, beta, r, p, se = stats.linregress(prediction.sel(time=slice('1900', '2010')), P_tot)
                    except ValueError:
                        r, p = 0, 1
                        print('Regression failed for lat {} lon {}'.format(precip_lat, precip_lon))

                    with open('CNN_models/reconstructions_piecewise.txt', 'a') as f: 
                        f.write('{} {} {} {}\n'.format(precip_lon, precip_lat, r, p))
