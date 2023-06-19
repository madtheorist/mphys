'''
A module for miscellaneous utility functions
Use test_env

Jesse Wang

'''

## Imports

import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt
import cartopy as cart
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import math
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import colors
from scipy import stats, optimize

## General functions for data manipulation/extraction

def map_with_latlon(lw_=2, central_longitude=0):
    '''
    Useful function to quickly make maps
    by Andrew Williams
    '''   
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    fig, ax = plt.subplots(dpi=100, subplot_kw={'projection':ccrs.PlateCarree(central_longitude=central_longitude)})

    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.coastlines(lw=lw_)
    ax.set_global()
    
    return fig, ax


def point_delta_precip_vector(delta_precip, mylat=0, mylon=0, time_avg='ANNUAL', weight_by_lat=True):
    '''
     Function to select a lat/lon point of delta precipitation data
     if weight_by_lat is true, we weight by cos(latitude)
    '''
    if weight_by_lat: 
        return delta_precip.sel(time_avg=time_avg, 
                         lat=mylat, 
                         lon=mylon).values * np.cos(np.deg2rad(mylat))
    else:
        return delta_precip.sel(time_avg=time_avg, 
                         lat=mylat, 
                         lon=mylon).values
    
    
def get_delta_precip_vector(delta_precip, lats, lons, time_avg='ANNUAL'):
    '''
    Given a set of lats and lons defining a box of gridpoints,
    Returns delta_precip vector averaged over the region.
    Weights the average by cos(latitude).
    '''
    delta_precips = delta_precip.sel(time_avg=time_avg, 
                                       lat=lats, 
                                       lon=lons)
    weights = np.cos(np.deg2rad(delta_precip.lat))
    weights.name = "weights" # Grid elements are larger near the equator
    delta_precips_weighted = delta_precips.weighted(weights)
    delta_precip_vector = delta_precips_weighted.mean(dim=["lon", "lat"])
    return delta_precip_vector


def visualise_point(mylon, mylat, central_longitude=0):
    '''
    Simple function to visualise a point on the world map
    '''
    fig, ax = map_with_latlon(lw_=1, central_longitude=central_longitude)
    plt.plot(mylon, mylat,  markersize=5, marker='o', color='red', 
             transform=ccrs.PlateCarree())
    plt.title('Location of precipitation point')
    plt.show()
    
## GTO-related

def calculate_GTO(delta_SST, delta_precip_vector, verbose=True,
                 lat_range = [-60, 60], lon_range =[0, 360], sig_test=True, return_GTO=True):
    '''
    Function to compute the GTO at each grid point
    delta_SST: xarray array with dimensions 5000 x 145 x 192
    delta_precip_vector: numpy array with 5000 elements
    sig_test: if True, sets values of statistically insignificant elements in G to NaN
    '''

    # Defining global variables
    T_max = 2 # maximum temperature perturbation
    a = 6371 # earth radius in kilometers
    phi_0 = math.radians(1.25) # latitude spacing in radians
    lambda_0 = math.radians(1.875) # longitude spacing in radians
    lats = delta_SST.lat.values
    lons = delta_SST.lon.values
    
    # temporary datarray to store value of GTO
    G = xr.DataArray(np.zeros((len(lats),len(lons))), coords=[lats, lons], dims=["lat", "lon"]) 
    L_y = a*phi_0
    n_ensemble = len(delta_precip_vector)
    
    for lat in lats:
        
        if lat < lat_range[0] or lat > lat_range[1]:
            continue
        L_x = a*lambda_0*math.cos(math.radians(lat))
        
        for lon in lons:
            if lon < lon_range[0] or lon > lon_range[1]:
                continue
            # Calculate the GTO at gridpoint
            cov = np.cov(delta_SST.sel(lat=lat, lon=lon).values, delta_precip_vector)[0,1 ] / ((1/3)*T_max**2 * L_x * L_y)
            
            # Check statistical significance
            # r is cov(delta_SST, delta_precip) divided by the product of the variances
            
            r = np.cov(delta_SST.sel(lat=lat, lon=lon).values,
                    delta_precip_vector)[0, 1] / np.sqrt(np.cov(delta_SST.sel(lat=lat, lon=lon).values, 
                             delta_precip_vector)[1, 1]*np.cov(delta_SST.sel(lat=lat, lon=lon).values,
                    delta_precip_vector)[0, 0])
            
            t = r*np.sqrt((n_ensemble-2)/(1-r**2))
            p = 1 - stats.norm.cdf(np.abs(t))
            is_significant = np.greater_equal(5, p*100*2)
            
            # If statistically significant, append to G matrix
            if is_significant:
                G.loc[lat, lon] = cov
            else:
                G.loc[lat, lon] = np.nan
            
        if verbose and (lat % 10 == 0):
            print('Latitude {} completed'.format(lat))

    if return_GTO:
        return G
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        G.to_netcdf("GTO_{}.nc".format(timestamp))
    


def calculate_GTO_piecewise(delta_SST, 
                            delta_precip_vector, 
                            verbose=True,
                            lat_range = [-60, 60], 
                            lon_range =[0, 360], 
                            sig_test=True, 
                            return_GTO=True):
    '''
    Function to compute the piecewise GTO at each grid point, 
    one for +ve delta_SST and one for -ve delta_SST
    delta_SST: xarray array with dimensions 5000 x 145 x 192
    delta_precip_vector: numpy array with 5000 elements
    sig_test: if True, sets values of statistically insignificant elements in G to NaN
    '''

    # Defining global variables
    T_max = 2 # maximum temperature perturbation
    a = 6371 # earth radius in kilometers
    phi_0 = math.radians(1.25) # latitude spacing in radians
    lambda_0 = math.radians(1.875) # longitude spacing in radians
    lats = delta_SST.lat.values
    lons = delta_SST.lon.values
    
    # temporary datarrays to store value of GTOs
    G_pos = xr.DataArray(np.zeros((len(lats),len(lons))), coords=[lats, lons], dims=["lat", "lon"])
    G_neg = xr.DataArray(np.zeros((len(lats),len(lons))), coords=[lats, lons], dims=["lat", "lon"])
    L_y = a*phi_0
    n_ensemble = len(delta_precip_vector)
    
    
    for lat in lats:
        
        if lat < lat_range[0] or lat > lat_range[1]:
            continue
            
        L_x = a*lambda_0*math.cos(math.radians(lat))
        
        for lon in lons:
            
            if lon < lon_range[0] or lon > lon_range[1]:
                continue
                
            # Calculate the GTOs at gridpoint
            
            pos_idx = np.argwhere(np.array(delta_SST.sel(lat=lat, lon=lon)) >= 0).flatten() # get idx for which delta_SST is positive
            neg_idx = np.argwhere(np.array(delta_SST.sel(lat=lat, lon=lon)) < 0).flatten() # get idx for which delta_SST is negative
            n_pos = len(pos_idx)
            n_neg = len(neg_idx)
            
            # Calculate positive and negative covariance
            pos_cov = np.cov(delta_SST.sel(lat=lat, lon=lon, idx=pos_idx).values, 
                         delta_precip_vector[pos_idx])[0,1] / ((1/3)*T_max**2 * L_x * L_y)
            neg_cov = np.cov(delta_SST.sel(lat=lat, lon=lon, idx=neg_idx).values, 
                         delta_precip_vector[neg_idx])[0,1] / ((1/3)*T_max**2 * L_x * L_y)
            
            # Check statistical significance for positive GTO
            r = np.cov(delta_SST.sel(lat=lat, lon=lon, idx=pos_idx).values,
                    delta_precip_vector[pos_idx])[0, 1] / np.sqrt(np.cov(delta_SST.sel(lat=lat, lon=lon, idx=pos_idx).values, 
                             delta_precip_vector[pos_idx])[1, 1]*np.cov(delta_SST.sel(lat=lat, lon=lon, idx=pos_idx).values,
                    delta_precip_vector[pos_idx])[0, 0])
            
            t = r*np.sqrt((n_pos-2)/(1-r**2))
            p = 1 - stats.norm.cdf(np.abs(t))
            is_significant = np.greater_equal(5, p*100*2)
            
            # If statistically significant, append to G matrix
            if is_significant:
                G_pos.loc[lat, lon] = pos_cov
            else:
                G_pos.loc[lat, lon] = np.nan
                
            # Check statistical significance for negative GTO
            r = np.cov(delta_SST.sel(lat=lat, lon=lon, idx=neg_idx).values,
                    delta_precip_vector[neg_idx])[0, 1] / np.sqrt(np.cov(delta_SST.sel(lat=lat, lon=lon, idx=neg_idx).values, 
                             delta_precip_vector[neg_idx])[1, 1]*np.cov(delta_SST.sel(lat=lat, lon=lon, idx=neg_idx).values,
                    delta_precip_vector[neg_idx])[0, 0])
            
            t = r*np.sqrt((n_neg-2)/(1-r**2))
            p = 1 - stats.norm.cdf(np.abs(t))
            is_significant = np.greater_equal(5, p*100*2)
            
            # If statistically significant, append to G matrix
            if is_significant:
                G_neg.loc[lat, lon] = neg_cov
            else:
                G_neg.loc[lat, lon] = np.nan
            
        if verbose and (lat % 10 == 0):
            print('Latitude {} completed'.format(lat))

    if return_GTO:
        return G_pos, G_neg
    
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        G.to_netcdf("GTO_{}.nc".format(timestamp))

    
def visualise_GTO(G, lats, lons, central_longitude=0, levels=10,
                 vmax=1e-6, vmin=-1e-6):
    '''
    Visualise the GTO on a contour map
    G: xarray dataarray
    Returns fig, ax objects
    '''

    G_, lons_ = add_cyclic_point(G, coord=lons) # avoids white bar down the middle

    fig, ax = map_with_latlon(central_longitude=central_longitude)
    fig.set_size_inches(10, 5)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    ax.add_feature(cfeature.LAND, zorder=2, edgecolor='k', facecolor='white')
    divnorm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)

    contourf = ax.contourf(lons_, lats, G_,
                    transform=ccrs.PlateCarree(),
                    cmap='RdBu_r', norm=divnorm, levels=levels)
    ax.contour(lons_, lats, G_,
                    transform=ccrs.PlateCarree(),
                    norm=divnorm, levels=levels,
                          linewidths=1, colors='black', zorder=1)
    cbar = fig.colorbar(contourf, location='bottom', pad = 0.1)
    cbar.set_label(r'GTO sensitivity ($\mathrm{mm} \ \mathrm{day}^{-1} \ \mathrm{km}^{-2} \ \mathrm{K}^{-1}$)')
    ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())
    
    return fig, ax


