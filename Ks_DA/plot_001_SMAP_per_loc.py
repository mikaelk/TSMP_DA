import numpy as np
import os
# import re
# import shutil
from datetime import datetime, timedelta
import pandas as pd
from glob import glob
import xarray as xr
# import netCDF4
import sys
from DA_operators import operator_clm_SMAP, operator_clm_FLX, haversine_distance
import matplotlib.pyplot as plt
import pickle

def save_dict_to_pickle(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)
    print(f"Dictionary saved to {filename}")

# 2. Load a pickle file back to a Python dict
def load_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    print(f"Dictionary loaded from {filename}")
    return loaded_dict


folder_results = '/p/scratch/cjibg36/kaandorp2/TSMP_results/eTSMP/DA_eCLM_cordex_444x432_v14_1y_iter1'
folder_storage = '/p/largedata/jibg36/kaandorp2/run_data/DA_eCLM_cordex_444x432_v14_1y_iter1'
thres_dx = 50
n_ensemble = 64
n_iter = 1
last_iter_ML_only = True #settings_DA['last_iter_ML_only']
# iterations_plot = [0,3,6,7] #0: OL, -1: n_iter
iterations_plot = [0,1] #0: OL, -1: n_iter

assert n_iter == iterations_plot[-1]

if folder_results not in sys.path:
    sys.path.insert(0,os.path.join(folder_results,'settings'))
from settings_copy import settings_run,settings_clm,settings_pfl,settings_sbatch,settings_DA,settings_gen,date_results_binned,freq_output,date_range_noleap

dir_figs = os.path.join(folder_results,'figures/01_SMAP_per_loc')
if not os.path.exists(dir_figs):
    print('Creating folder to store FLX information: %s' % (dir_figs) )
    os.makedirs(dir_figs)
    
eCLM_dz = np.array([0.02,0.04,0.06,0.08,0.120,
                    0.160,0.200,0.240,0.280,0.320,
                    0.360,0.400,0.440,0.540,0.640,
                    0.740,0.840,0.940,1.040,1.140,
                    2.390,4.676,7.635,11.140,15.115])


i_date = 0
date_results_iter = date_results_binned[i_date].copy()
date_start_sim = date_results_binned[i_date][0][0]#datetime(2019,1,2,12,0,0)
date_end_sim = date_results_binned[i_date][-1][-1]#datetime(2019,12,31,12,0,0)

# dir_setup = settings_run['dir_setup']
dir_setup = folder_storage
str_date = str(date_start_sim.date()).replace('-','') + '-' + str(date_end_sim.date()).replace('-','')
dir_date = os.path.join(dir_setup,str_date)


operator = {}
data_measured = {}
data_latlon = {}
data_var = {}
data_date = {}

operator['FLX'] = operator_clm_FLX(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_FLX'],ignore_rivers=False)
data_measured['FLX'],data_var['FLX'],data_latlon['FLX'] = operator['FLX'].get_measurements(date_results_iter,
                                                                                           date_DA_start=date_start_sim,
                                                                                           return_latlon=True)
# # Operator for FLUXNET
# operator['FLX'] = operator_clm_FLX(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_FLX'],ignore_rivers=False)
# data_measured['FLX'],data_var['FLX'],data_latlon['FLX'] = operator['FLX'].get_measurements(date_results_iter,
#                                                                                            date_DA_start=date_start_sim,
#                                                                                            return_latlon=True)
# # Operator for SMAP
# operator['SMAP'] = operator_clm_SMAP(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_SMAP'],ignore_rivers=False)
# data_measured['SMAP'],_,data_latlon['SMAP'],data_date['SMAP'] = operator['SMAP'].get_measurements(date_results_iter,
#                                                                                                   date_DA_start=date_start_sim,
#                                                                                                   return_latlon=True,return_date=True)
station_names = list(operator['FLX'].data_flx.keys())[0:-30]

for station_name in station_names:
# station_name = 'DE-Obe'
    print('----------------------------------------')
    print('------Calculating station %s---------' % station_name)
    print('----------------------------------------')
    
    pickle_filename = os.path.join(dir_figs,'%s.pickle'%station_name)

    if os.path.exists(pickle_filename):
        results = load_dict_from_pickle(pickle_filename)
        results_clm = results['results_clm'] 
        results_smap = results['results_smap']
        date_results_iter = results['date_results_iter']
        station_name = results['station_name'] 
        lon_station = results['lon_station']
        lat_station = results['lat_station']
        n_iter = results['n_iter']
        i_OL = results['i_OL']
    else:
        date_results_iter = date_results_binned[i_date].copy()

        # Operator for FLUXNET
        operator['FLX'] = operator_clm_FLX(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_FLX'],ignore_rivers=False)
        data_measured['FLX'],data_var['FLX'],data_latlon['FLX'] = operator['FLX'].get_measurements(date_results_iter,
                                                                                                   date_DA_start=date_start_sim,
                                                                                                   return_latlon=True)
        # Operator for SMAP
        operator['SMAP'] = operator_clm_SMAP(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_SMAP'],ignore_rivers=False)
        data_measured['SMAP'],_,data_latlon['SMAP'],data_date['SMAP'] = operator['SMAP'].get_measurements(date_results_iter,
                                                                                                          date_DA_start=date_start_sim,
                                                                                                          return_latlon=True,return_date=True)

        lon_station = operator['FLX'].data_flx[station_name]['lon']
        lat_station = operator['FLX'].data_flx[station_name]['lat']

        all_dates = date_results_iter[0]

        results_smap = {}
        results_clm = {}

        for i_iter in iterations_plot:
            str_iter = 'i%3.3i' % i_iter
            dir_iter = os.path.join(dir_date,str_iter)
            settings_run['dir_iter'] = dir_iter

            if i_iter == 0:
                i_real = 0
                _ = operator['SMAP'].interpolate_model_results(i_real,settings_run,indices_z=[0,1],var='SOILLIQ')#[data_mask['SMAP']]

                # 0) find closest CLM location (i_closest_clm)
                i_min = np.nanargmin(haversine_distance((lat_station,lon_station),np.array([operator['SMAP'].grid_TSMP.lat_centre.values.ravel(),operator['SMAP'].grid_TSMP.lon_centre.values.ravel()]).T ) )
                lat_imin = operator['SMAP'].grid_TSMP.lat_centre.values.ravel()[i_min]
                lon_imin = operator['SMAP'].grid_TSMP.lon_centre.values.ravel()[i_min]
                i_min_grid = np.where(operator['SMAP'].grid_TSMP.lat_centre.values == lat_imin)
                assert(i_min_grid == np.where(operator['SMAP'].grid_TSMP.lon_centre.values == lon_imin))

                i_lat_min = i_min_grid[0][0]
                i_lon_min = i_min_grid[1][0]

            if i_iter == n_iter:
                #validation run
                date_validation_start = date_results_iter[-1][-1]
                date_results_iter.append(list(date_range_noleap(date_validation_start,
                                                                date_validation_start+timedelta(days=settings_run['ndays_validation']),freq=freq_output)))
                data_measured['SMAP'],_,data_latlon['SMAP'],data_date['SMAP'] = operator['SMAP'].get_measurements(date_results_iter,
                                                                                                                  date_DA_start=date_start_sim,
                                                                                                                  return_latlon=True,
                                                                                                                  return_date=True)
                i_real = 0
                _ = operator['SMAP'].interpolate_model_results(i_real,settings_run,indices_z=[0,1],var='SOILLIQ')#[data_mask['SMAP']]

                all_dates = np.unique(np.array([date_results_iter[i][i2] for i in range(2) for i2 in range(len(date_results_iter[i]))]))

            results_smap[i_iter] = np.nan*np.ones([1,len(all_dates)])
            results_clm[i_iter] = np.nan*np.ones([n_ensemble+2,len(all_dates)]) #plus 2: +OL, +ML


            # 1) 
            for i_date_,date_ in enumerate(all_dates):
                print(date_)
                if date_ in list(operator['SMAP'].sm_out.keys()):
                    mask_date = data_date['SMAP']== date_
                    dist_ = haversine_distance((lat_station,lon_station),
                                                                data_latlon['SMAP'][mask_date] ) 
                    i_closest = np.nanargmin(dist_)
                    if dist_[i_closest] < thres_dx:
                        results_smap[i_iter][0,i_date_] = data_measured['SMAP'][mask_date][i_closest]
                    else:
                        results_smap[i_iter][0,i_date_] = np.nan
                else:
                    results_smap[i_iter][0,i_date_] = np.nan

                # 2)
                for folder_real in sorted(glob(os.path.join(dir_iter,'*'))):
                    if '_old' in folder_real:
                        pass
                    else:
                        i_real = int(os.path.basename(folder_real)[1:])

                        key_0 = list(operator['SMAP'].sm_out.keys())[0]
                        date_0 = str(list(operator['SMAP'].sm_out.keys())[0])[0:10]
                        if date_ in list(operator['SMAP'].files_clm.keys()):
                            key_0 = date_#list(operator['SMAP'].sm_out.keys())[i_date_]
                            date_0 = str(date_)[0:10]

                        file_clm = operator['SMAP'].files_clm[key_0].replace('i000','i%3.3i'%i_iter).replace('R000','R%3.3i'%i_real).replace(date_0,str(date_)[0:10])
                        # print(file_clm)
                        data_clm = xr.open_dataset(file_clm)

                        SM = data_clm.SOILLIQ[0]/(eCLM_dz[0:20,np.newaxis,np.newaxis]*1000)
                        results_clm[i_iter][i_real,i_date_] = SM[np.array([0,1]),i_lat_min,i_lon_min].values.mean()

                        i_OL = i_real #save last realization number; this is the OL run

        results = {}
        results['results_clm'] = results_clm
        results['results_smap'] = results_smap
        results['date_results_iter'] = date_results_iter
        results['station_name'] = station_name
        results['lon_station']=lon_station
        results['lat_station']=lat_station
        results['n_iter']=n_iter
        results['i_OL']=i_OL

        save_dict_to_pickle(results,pickle_filename)


    # Finally plotting
    dates = date_results_iter[0]
    all_dates = np.unique(np.array([date_results_iter[i][i2] for i in range(2) for i2 in range(len(date_results_iter[i]))]))
    minval = np.nanmin([np.nanmin(results_clm[key_]) for key_ in results_clm.keys()])
    maxval = np.nanmax([np.nanmax(results_clm[key_]) for key_ in results_clm.keys()])

    hatches = ['/', '\\', '|', '-']*3

    cmap = plt.cm.tab10
    fig,ax = plt.subplots(1,figsize=(12,8))
    ax.set_ylim([.9*minval,1.1*maxval])
    ax.plot(all_dates,results_smap[n_iter][0,:],'kx',label='SMAP')

    for i_iter in iterations_plot[:-1]:
        ax.fill_between(dates, np.nanmin(results_clm[i_iter],axis=0), np.nanmax(results_clm[i_iter],axis=0), color=cmap(i_iter),hatch=hatches[i_iter],alpha=.2,label='iter. %i'%i_iter)

    ax.plot(all_dates,results_clm[n_iter][i_OL,:],':',color=cmap(0),label='OL')
    ax.plot(all_dates,results_clm[n_iter][0,:],':',color=cmap(3),label='ML, iter. %i'%n_iter)
    ax.legend()
    ax.set_ylabel(r'Soil moisture [m$^3$/m$^3$]')
    ax.set_title('%s (lat: %3.2f, lon: %3.2f)' % (station_name,lat_station,lon_station))
    ax.plot([date_results_iter[0][-1],date_results_iter[0][-1]],[0.9*minval,1.1*maxval],'k--',alpha=.8)
    fig.savefig(os.path.join(dir_figs,'%s.pdf'%station_name))
    fig.savefig(os.path.join(dir_figs,'%s.png'%station_name))