'''
A script for reconstructions

Use the conda environment 'xesmf_env' for this.

Jesse Wang

'''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import glob
import regionmask # need to install this (dependent on geopandas) using pip, AFTER everything else!
import xesmf as xe
from scipy import stats
import argparse
import warnings

def preprocess_GTO(filename):
    '''
    Masks out land and replaces with NaN
    '''
    G = xr.open_dataarray(filename)
    land=regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(G.lon.values, G.lat.values)
    G_processed = G.where(abs(G)>0).where(np.isnan(land), np.nan)
    return G_processed

def regrid_to_target(ds, ds_target, method='bilinear', save_weights=False):
    """
    Regrid from rectilinear grid to common grid
    
    Bilinear and conservative should be the most commonly used methods. 
    They are both monotonic (i.e. will not create new maximum/minimum). 
    """
    regridder = xe.Regridder(ds, ds_target, method, periodic=True, reuse_weights=save_weights)
    return regridder(ds)

def plot_comparison(observed_mean, prediction_scaled):
    '''
    Plot reconstruction against observed
    '''
    fig, ax = plt.subplots(figsize=(10,5))
    plt.subplots_adjust(bottom=0.15)
    observed_mean.plot(ax=ax, label='ERA-20C reanalysis data', linestyle='dashed')
    prediction_scaled.sel(time=slice('1900', '2010')).plot(ax=ax, label='Linear GTO reconstruction', color='red')
    ax.set_ylabel('Total precipitation at grid point (m)')
    ax.set_xlabel('Year')
    ax.legend()
    ax.set_title('r={}'.format(round(r,3)))  

if __name__ == '__main__':
    
    warnings.filterwarnings('ignore') # filter out deprecation warnings for now
    
    parser = argparse.ArgumentParser(
                    prog = 'reconstruction',
                    description = 'Calculates the precip reconstruction for user-specific point of the lat-lon grid and regresses with ERA-20C reanalysis')
    parser.add_argument('--lat', type=float, help='10 must be entered as 10.0')
    parser.add_argument('--lon', type=float)
    parser.add_argument('--fig', action='store_true', help='If true, generates reconstruction plots')
    args = parser.parse_args()
    mylat, mylon = args.lat, args.lon
    fig = args.fig
 
    lat_filename = str(mylat).replace(".","")
    lon_filename = str(mylon).replace(".","")
    
    filename = "GTO_data_10deg/GTO_lat{}_lon{}.nc".format(lat_filename, lon_filename)
  
    
    # mask out land and turn zeros into NaNs
    G = preprocess_GTO(filename)
    
    # Open SST dataset and sample by annual mean
    hadsst = xr.open_dataset("/gws/nopw/j04/aopp/andreww/CPDN/data/HadISST_sst.nc")['sst'].rename({'latitude':'lat', 'longitude':'lon'})
    hadsst = hadsst.where(hadsst>0, 0)
    hadsst_annual_mean = hadsst.resample(time='1Y').mean('time')
    
    # regridding to match reanalysis dataset
    ds_target = xr.Dataset({'lat': (['lat'], hadsst_annual_mean.lat.data),
                        'lon': (['lon'], hadsst_annual_mean.lon.data),})
    
    G_fixed = G.assign_coords(lon=(((G.lon - 180) % 360) - 180)).sortby('lon')
    G_fixed['lat'].attrs=hadsst_annual_mean['lat'].attrs
    G_fixed['lon'].attrs=hadsst_annual_mean['lon'].attrs
    G_regridded = regrid_to_target(ds=G_fixed, ds_target=ds_target)
    
    # making prediction
    weights = np.cos(np.deg2rad(G_regridded.lat))* 6371**2 * 1 * 1 * (np.pi/180)**2 * G_regridded
    product = hadsst_annual_mean.weighted(weights.fillna(0))
    prediction = product.sum(('lon', 'lat'), skipna=True)
    
    # comparing with ERA-20C
    observed_mean = xr.open_dataarray("observed_mean_annual.nc").sel(lat=mylat, lon=mylon, method='nearest').compute()
    
    alpha, beta, r, p, se = stats.linregress(prediction.sel(time=slice('1900', '2010')), observed_mean)
    prediction_scaled = alpha * prediction + beta
    
    print(mylon, mylat, r, p)
    
    if fig:
        plot_comparison(observed_mean, prediction_scaled)
        plt.savefig('reconstruction_lat{}_lon{}.pdf'.format(lat_filename, lon_filename))
   
   

    
    
    