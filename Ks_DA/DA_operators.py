import numpy as np
import os
import shutil
from datetime import datetime, timedelta
import pandas as pd
from glob import glob
import xarray as xr
import netCDF4
from scipy.interpolate import griddata

def get_TSMP_grid(file_centre,file_corner):
    grid_centre = xr.open_dataset(file_centre, decode_times=False)
    grid_corner = xr.open_dataset(file_corner)

    lon_edges = np.concatenate((grid_corner.lon.values[:,-1],grid_corner.lon.values[:,0],grid_corner.lon.values[0,:],grid_corner.lon.values[-1,:]))
    lat_edges = np.concatenate((grid_corner.lat.values[:,-1],grid_corner.lat.values[:,0],grid_corner.lat.values[0,:],grid_corner.lat.values[-1,:]))
    lsm_edges = np.zeros(lat_edges.shape)
    
    grid_TSMP = xr.Dataset(data_vars=dict(lon_centre=(['x','y'], grid_centre.lon.values),
                                          lat_centre=(['x','y'], grid_centre.lat.values),
                                          lon_corner=(['xc','yc'], grid_corner.lon.values),
                                          lat_corner=(['xc','yc'], grid_corner.lat.values),
                                          lsm=(['x','y'], grid_centre.LLSM[0].values),
                                          lon_edges=(['ne'], lon_edges),
                                          lat_edges=(['ne'], lat_edges),
                                          lsm_edges=(['ne'], lsm_edges)))
    
    return grid_TSMP

def get_TSMP_coords(file_centre):
    grid_centre = xr.open_dataset(file_centre, decode_times=False)
    grid_TSMP = xr.Dataset(data_vars=dict(lon_centre=(['x','y'], grid_centre.lon.values),
                                          lat_centre=(['x','y'], grid_centre.lat.values),
                                          lsm=(['x','y'], grid_centre.LLSM[0].values)))
    return grid_TSMP
    
def parse_datetime_clm(files):
    dates = []
    for file_ in files:
        date_ = pd.to_datetime(os.path.basename(file_).split('h0.')[1].split('.nc')[0][0:10]) + timedelta(seconds=int(os.path.basename(file_).split('h0.')[1].split('.nc')[0][11:]))
        dates.append(date_)
    return dates
    
    
def parse_datetime_smap(files):
    dates = []
    for file_ in files:
        date_ = pd.to_datetime(os.path.basename(file_).split('P_E_')[1].split('_')[0]) + timedelta(hours=12)
        dates.append(date_)
    return dates

def read_SMAP(file_):
    ncf = netCDF4.Dataset(file_)
    group_pm = ncf.groups.get('Soil_Moisture_Retrieval_Data_PM')
    group_am = ncf.groups.get('Soil_Moisture_Retrieval_Data_AM')

    data_pm = xr.open_dataset(xr.backends.NetCDF4DataStore(group_pm))
    data_am = xr.open_dataset(xr.backends.NetCDF4DataStore(group_am))

    return data_am, data_pm


class operator_clm_SMAP:
    
    def __init__(self,file_lsm,file_corner,folder_SMAP):
   
        # 1) get TSMP grid: is used to select SMAP results falling withing this domain. A corner file is required to discard points outside of the domain       
        self.grid_TSMP = get_TSMP_grid(file_lsm,file_corner)   
        self.folder_SMAP = folder_SMAP
        
    def flatten_y(self,y_in):
        # flatten dict with values into a np.array
        y_out = np.array([])
        for val in y_in.values():
            y_out = np.append(y_out,val)        
        return y_out
        
    def get_measurements(self,date_results_iter):
        self.lons_out = {}
        self.lats_out = {}
        self.sm_out = {}


        for date_results in date_results_iter:
            for date_ in date_results[1:]:

                # 2) get all available SMAP dates
                files_SMAP = sorted(glob(os.path.join(self.folder_SMAP,'nc_%4.4i/*.nc'%date_.year)))
                dates_SMAP = parse_datetime_smap(files_SMAP)

                # 3) for the given model date (date_) get the corresponding SMAP data
                i_closest = np.argmin(np.abs(pd.to_datetime(dates_SMAP) - date_))
                assert abs(dates_SMAP[i_closest] - date_) < timedelta(days=1), 'Measurement and model output differ more than a day, TO DO: implement if statement to avoid'
                file_SMAP = files_SMAP[i_closest]
                data_SMAP_am, data_SMAP_pm = read_SMAP(file_SMAP)

                # 4) select valid datapoints where the soil moisture and location are given
                mask_data_valid = ~np.isnan(data_SMAP_am.soil_moisture) & ~np.isnan(data_SMAP_am.longitude) 

                # 5) Select valid points: valid data, and within the given TSMP domain
                sm = data_SMAP_am.soil_moisture.where(mask_data_valid).values[mask_data_valid]
                lons = data_SMAP_am.longitude.where(mask_data_valid).values[mask_data_valid]
                lats = data_SMAP_am.latitude.where(mask_data_valid).values[mask_data_valid]

                # poor man's check that measurement points are on land and lie within the TSMP domain... There might be a better solution for this (convex hull check on EU-11 polygon?) 
                mask_SMAP_lsm = griddata((np.concatenate((self.grid_TSMP.lon_centre.values.ravel(),self.grid_TSMP.lon_edges)),np.concatenate((self.grid_TSMP.lat_centre.values.ravel(),self.grid_TSMP.lat_edges))),
                                         np.concatenate((self.grid_TSMP.lsm.values.ravel(),self.grid_TSMP.lsm_edges)),
                                         (lons,lats),method='nearest') == 2

                self.lons_out[date_] = lons[mask_SMAP_lsm]
                self.lats_out[date_] = lats[mask_SMAP_lsm]
                self.sm_out[date_] = sm[mask_SMAP_lsm]
        return self.flatten_y(self.sm_out)
                
    def interpolate_model_results(self,i_real,settings_run):
        self.data_TSMP_i = {}
        files_clm = sorted(glob(os.path.join(settings_run['dir_iter'],'R%3.3i/**/clm.clm2.h0.*.nc'%i_real)))
        assert len(files_clm) == len(self.sm_out.keys()), 'Something might have gone wrong in realization %i: not every date has a matching file'%i_real

        for i1,date_ in enumerate(self.sm_out.keys()):
            file_clm = sorted(glob(os.path.join(settings_run['dir_iter'],'R%3.3i/**/clm.clm2.h0.*.nc'%i_real)))[i1]
            # print(date_,file_clm)

            data_TSMP = xr.open_dataset(file_clm)
            # add curvilinear lon/lat 
            data_TSMP = data_TSMP.assign_coords(lon_c=(('lat','lon'), self.grid_TSMP.lon_centre.values))
            data_TSMP = data_TSMP.assign_coords(lat_c=(('lat','lon'), self.grid_TSMP.lat_centre.values))

            self.data_TSMP_i[date_] = griddata((data_TSMP.lon_c.values.ravel(),data_TSMP.lat_c.values.ravel()),data_TSMP.H2OSOI[0,0].values.ravel(),(self.lons_out[date_],self.lats_out[date_]), method='nearest')

        # save most likely parameter output for plotting
        if i_real == 0:
            self.data_TSMP_ml = self.data_TSMP_i.copy()
            
        return self.flatten_y(self.data_TSMP_i)
