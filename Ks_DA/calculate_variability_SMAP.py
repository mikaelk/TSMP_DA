import numpy as np
import os
import re
import shutil
from datetime import datetime, timedelta
import pandas as pd
from glob import glob
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import time
import pickle

def haversine_distance(loc1, loc_array):
    """
    Calculate the Haversine distance between a point and an array of points on the Earth
    given their latitude and longitude in decimal degrees.

    Parameters:
    - loc1: Tuple containing the latitude and longitude of the first point (in decimal degrees).
    - loc_array: Array of tuples, each containing the latitude and longitude of a point (in decimal degrees).

    Returns:
    - Array of distances between loc1 and each point in loc_array (in kilometers).
    """
    if np.isnan(loc1[0]) and np.isnan(loc1[1]):
        distances = np.zeros(len(loc_array))
    else:
        # Radius of the Earth in kilometers
        R = 6371.0

        # Convert decimal degrees to radians
        lat1_rad, lon1_rad = np.radians(loc1)
        lat2_rad, lon2_rad = np.radians(np.array(loc_array).T)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distances = R * c
    return distances

from DA_operators import operator_clm_SMAP, operator_clm_FLX
from settings import settings_run,settings_clm,settings_pfl,settings_sbatch,settings_DA,settings_gen,date_results_binned,freq_output,date_range_noleap

pickle_filename = 'variability_calculations/variability_SMAP_0-40cells.pkl'
x_lower = np.linspace(0,12.5*40,1+(10))
x_upper = x_lower + 12.5
print(x_lower)
print(x_upper)

date_results_iter = [pd.date_range(datetime(2019,1,1),datetime(2019,12,31),freq='7d')]
operator = {}
data_measured = {}
data_latlon = {}
data_dates = {}
operator['SMAP'] = operator_clm_SMAP(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_SMAP'],ignore_rivers=False)
data_measured['SMAP_PM'],_,data_latlon['SMAP_PM'],data_dates['SMAP_PM'] = operator['SMAP'].get_measurements(date_results_iter,return_latlon=True,return_date=True,mode='pm')
data_measured['SMAP_AM'],_,data_latlon['SMAP_AM'],data_dates['SMAP_AM'] = operator['SMAP'].get_measurements(date_results_iter,return_latlon=True,return_date=True,mode='am')

# %matplotlib widget
results_all_dx = {}

for i3,(x_low_,x_upp_) in enumerate(zip(x_lower,x_upper)):
    
    x_mean = .5*(x_upp_ + x_low_)

    results_all_dx[x_mean] = {}

    c=0
    results = results_all_dx[x_mean]
    results['monthly'] = {}
    results['yearly'] = {}

    results['monthly']['n_tot'] = 12*[0]
    results['monthly']['diff_sq'] = 12*[0]
    
    if i3 == 0:
        results['monthly']['pairs_x'] = [[] for i in range(12)] 
        results['monthly']['pairs_y'] = [[] for i in range(12)]

    results['yearly']['n_tot'] = 0
    results['yearly']['diff_sq'] = 0

    for i2,date_ in enumerate(date_results_iter[0]):
        mask_t1 = (np.array(data_dates['SMAP_AM']) == date_)
        mask_t2 = (np.array(data_dates['SMAP_PM']) == date_)

        array_latlon1 = data_latlon['SMAP_AM'][mask_t1]
        array_latlon2 = data_latlon['SMAP_PM'][mask_t2]

        array_val1 = data_measured['SMAP_AM'][mask_t1]
        array_val2 = data_measured['SMAP_PM'][mask_t2]

        i_month = date_.month - 1


        for i1,((lat_, lon_),sm_) in enumerate(zip(array_latlon1,array_val1)):
        # lat_,lon_ = array_latlon1[2,:]

            dx = haversine_distance((lat_,lon_),array_latlon2)

            mask = (dx < x_upp_) & (dx >= x_low_)

            if mask.sum() > 0:
                # print(i1,dx,mask.sum())
                # break

                dSM = sm_ - array_val2[mask]

                results['monthly']['n_tot'][i_month] += mask.sum()
                results['monthly']['diff_sq'][i_month] += np.sum(dSM**2)
                if i3 == 0:
                    results['monthly']['pairs_x'][i_month].extend([sm_]*len(dSM))
                    results['monthly']['pairs_y'][i_month].extend(list(array_val2[mask]))

                results['yearly']['n_tot'] += mask.sum()
                results['yearly']['diff_sq'] += np.sum(dSM**2)


                c+=1

            if i1%2000 == 0:
                print('%i/%i' % (i1,len(array_latlon1)) )
        print('i2: %i/%i' % (i2,len(date_results_iter[0])) )
        
    print('------------length scale done---------------')
    print(x_low_,x_upp_)
    print('------------length scale done---------------')

try:
    with open(pickle_filename, 'wb') as pickle_file:
        pickle.dump(results_all_dx, pickle_file)
except:
    with open('SMAP_var.pkl', 'wb') as pickle_file:
        pickle.dump(results_all_dx, pickle_file)