'''
A script for reconstructions with basic CNN with very roughly tuned hyperparameters.

Use the conda environment 'xesmf_env' for this.

Jesse Wang

'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks, optimizers
import xarray as xr
import xesmf as xe
from scipy import stats
from sklearn.preprocessing import StandardScaler
import regionmask
import argparse
import warnings
import time

warnings.filterwarnings("ignore")

def preprocess_data(delta_SST, delta_precip, precip_lat, precip_lon, train_percentage=0.8, scale_Y=True):
    '''
    Get X_train, X_test, Y_train, Y_test (numpy arrays) from delta_SST and delta_precip.
    
    Used to train and evaluate model.
    
    '''
    
    # need to mask out land
    landmask = np.isnan(regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(delta_SST.lon.values, delta_SST.lat.values))
    
    # X is in range from (-2, 2), so fairly reasonable and no need to feature scale.
    X = delta_SST.sel(lat=slice(60, -60)).where(landmask).to_numpy()
    X = np.nan_to_num(X[..., np.newaxis]) # required third dimension for input to CNN. NaN values become zero.
    
    Y = delta_precip.sel(time_avg='ANNUAL', lat=precip_lat, lon=precip_lon).to_numpy()
    Y = Y[..., np.newaxis]
    
    # Scale Y to unit variance and zero mean.
    if scale_Y:
        Y_scaler = StandardScaler()
        Y = Y_scaler.fit_transform(Y)
    
    num_examples = X.shape[0]
    
    X_train = X[:int(train_percentage*num_examples)]
    X_test = X[int(train_percentage*num_examples):]
    Y_train = Y[:int(train_percentage*num_examples)]
    Y_test = Y[int(train_percentage*num_examples):]
    
    return X_train, X_test, Y_train, Y_test


def create_model(lr=1e-3, 
                 optimizer='Adam', 
                 kernel_size=3, 
                 pooling_kernel_size=2, 
                 n_layers=3,
                 n_filters=16, 
                 n_dense_neurons=32,
                 dropout=True):
    
    '''
    Returns instance of Keras model, consisting of Convolutional+Max pooling layers fed into a fully-connected neural network.
    
    Default hyperparameters as given above seem to work well for the lat/lon points that I tested.
    '''
    
    model = models.Sequential()
    
    # n_layers refers to number convolutional + max pooling layers
    for i in range(n_layers):
        model.add(layers.Conv2D((2**i * n_filters), kernel_size, activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
  
    # flatten and feed into fully-connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(n_dense_neurons, activation='relu')) # highly sensitive
    if dropout:
        model.add(layers.Dropout(0.1))
    model.add(layers.Dense(16, activation='relu'))
    
    # regress
    model.add(layers.Dense(1, activation='linear'))
    
    if optimizer == 'Adam':
        optimizer = optimizers.Adam(learning_rate=lr)
    elif optimizer == 'SGD':
        optimizer = optimizers.SGD(learning_rate=lr)
        
    model.compile(loss='mean_squared_error', # one may use 'mean_absolute_error' as  mean_squared_error
                  optimizer=optimizer)
    
    return model


if __name__ == '__main__':
    
    t0 = time.time()
    
    parser = argparse.ArgumentParser(
                    prog = 'reconstruction',
                    description = 'Calculates the precip reconstruction for user-specific point of the lat-lon grid and regresses with ERA-20C reanalysis')
    parser.add_argument('--lat', type=float)
    parser.add_argument('--lon', type=float)
    args = parser.parse_args()
    precip_lat, precip_lon = args.lat, args.lon
    
    # Define precipitation gridpoint and batch size/max number of epochs.
    batch_size = 4
    epochs = 30
    validation = True # Generates r values for Corr(delta_preci predictions on val set, delta_precip from simulations)
    reconstruct = False # Performs time series reconstruction and generates r values
    
    print("Training Lat {} Lon {}".format(precip_lat, precip_lon))
    
    # Preprocess data
    delta_SST = xr.open_dataarray("delta_SST_pr.nc").compute()
    delta_precip = xr.open_dataarray("delta_precip_10deg.nc").compute()
    X_train, X_test, Y_train, Y_test = preprocess_data(delta_SST, delta_precip, precip_lat, precip_lon, scale_Y=True)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    
    # Define CNN architecture and train model
    model = create_model(lr=1e-3, optimizer='Adam', kernel_size=3, n_layers=3, n_filters=16)
    
    early_stopping_cb = callbacks.EarlyStopping(patience=5, 
                                            restore_best_weights=True)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2,
              validation_data=(X_test, Y_test), callbacks=[early_stopping_cb])
    
    t1 = time.time()
    print("Time to train model: {}".format(t1-t0))
    
    if validation:
        preds = model.predict(X_test)
        preds = preds[:,0]
        alpha_, beta_, r_, p_, se_ = stats.linregress(Y_test.flatten(), preds)
        
        with open('CNN_models/validation_r_values.txt', 'a') as f: 
            f.write('{} {} {} {}\n'.format(precip_lon, precip_lat, r_, p_))
    
    # Compare with ERA-20C reanalysis
    if reconstruct:
        observed_delta_sst = np.load('CNN_models/observed_delta_SST_1980-99.npy')
        deltaP_predicted = model.predict(observed_delta_sst).flatten()[31:142] #year range 1900-2010 

        observed_precip = xr.open_dataarray("observed_mean_annual.nc").compute()
        P_tot = observed_precip.sel(lat=precip_lat, lon=precip_lon).values
        #P_tot = P_tot[-21:] #year range 1990-2010 
        alpha, beta, r, p, se = stats.linregress(deltaP_predicted, P_tot)
        P_predicted = alpha * deltaP_predicted + beta

        # Write reconstruction accuracy (r value) and significance (p value) to file
        with open('CNN_models/reconstructions_cnn_20yr.txt', 'a') as f: 
            f.write('{} {} {} {}\n'.format(precip_lon, precip_lat, r, p))