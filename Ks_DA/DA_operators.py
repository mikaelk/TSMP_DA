import numpy as np
import os
import shutil
from datetime import datetime, timedelta
import pandas as pd
from glob import glob
import xarray as xr
import netCDF4
from scipy.interpolate import griddata
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def get_TSMP_grid(file_centre,file_corner,ignore_rivers=False):
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
    
    
    if ignore_rivers:
        # in the land-sea mask, set rivers to a value of 3, in which case they are ignored later in the interpolation process
        grid_TSMP['lsm'] = grid_TSMP['lsm'].where((grid_centre.mask_rivers.values==0),3)

    return grid_TSMP

def get_TSMP_coords(file_centre):
    grid_centre = xr.open_dataset(file_centre, decode_times=False)
    grid_TSMP = xr.Dataset(data_vars=dict(lon_centre=(['x','y'], grid_centre.lon.values),
                                          lat_centre=(['x','y'], grid_centre.lat.values),
                                          lsm=(['x','y'], grid_centre.LLSM[0].values)))
    return grid_TSMP
    
# def parse_datetime_clm(files):
#     dates = []
#     for file_ in files:
#         date_ = pd.to_datetime(os.path.basename(file_).split('h0.')[1].split('.nc')[0][0:10]) + timedelta(seconds=int(os.path.basename(file_).split('h0.')[1].split('.nc')[0][11:]))
#         dates.append(date_)
#     return dates
    
    
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

def unpackbits(x, num_bits):
    #function that converts int into series of bits (used for 0/1 in SMAP quality flags)
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

class operator_clm_SMAP:
    
    def __init__(self,file_lsm,file_corner,folder_SMAP,ignore_rivers=False):
   
        # 1) get TSMP grid: is used to select SMAP results falling withing this domain. A corner file is required to discard points outside of the domain       
        self.grid_TSMP = get_TSMP_grid(file_lsm,file_corner,ignore_rivers=ignore_rivers)   
        self.folder_SMAP = folder_SMAP
        
    def flatten_y(self,y_in):
        # flatten dict with values into a np.array
        y_out = np.array([])
        for val in y_in.values():
            y_out = np.append(y_out,val)        
        return y_out
        
    def get_measurements(self,date_results_iter,date_DA_start=datetime(1900,1,1),date_DA_end=datetime(2200,1,1),
                         mode='pm',qual_level=1):
        self.lons_out = {}
        self.lats_out = {}
        self.sm_out = {}

        for date_results in date_results_iter:
            for date_ in date_results[1:]:

                if date_ < date_DA_start or date_ > date_DA_end:
                    print('Skipping %s: outside of range %s-%s'%(str(date_),str(date_DA_start),str(date_DA_end)) )
                else:
                    # 2) get all available SMAP dates
                    files_SMAP = sorted(glob(os.path.join(self.folder_SMAP,'nc_%4.4i/*.nc'%date_.year)))
                    dates_SMAP = parse_datetime_smap(files_SMAP)

                    # 3) for the given model date (date_) get the corresponding SMAP data
                    i_closest = np.argmin(np.abs(pd.to_datetime(dates_SMAP) - date_))
                    diff_days = abs(dates_SMAP[i_closest] - date_)
                    if diff_days > timedelta(days=1):
                        print('Measurement and model output differ more than a day, SMAP: %s, TSMP: %s'%(dates_SMAP[i_closest],date_))
                        # This happens e.g. when not every day has a measurement
                    else:
                        file_SMAP = files_SMAP[i_closest]
                        data_SMAP_am, data_SMAP_pm = read_SMAP(file_SMAP)
                        
                        if mode == 'am':
                            # 4) select valid datapoints where the soil moisture and location are given
                            mask_data_valid = ~np.isnan(data_SMAP_am.soil_moisture) & ~np.isnan(data_SMAP_am.longitude) 
                            if qual_level > 0: #quality level; when 0 use all retrievals, with 1 only use recommended retrievals
                                qual_flags = unpackbits(data_SMAP_am.retrieval_qual_flag.values.astype(int),4)
                                mask_data_valid = mask_data_valid & (qual_flags[:,:,0]==0)
                            # 5) Select valid points: valid data, and within the given TSMP domain
                            sm = data_SMAP_am.soil_moisture.where(mask_data_valid).values[mask_data_valid]
                            lons = data_SMAP_am.longitude.where(mask_data_valid).values[mask_data_valid]
                            lats = data_SMAP_am.latitude.where(mask_data_valid).values[mask_data_valid]
                        elif mode == 'pm':
                            mask_data_valid = ~np.isnan(data_SMAP_pm.soil_moisture_pm) & ~np.isnan(data_SMAP_pm.longitude_pm) 
                            if qual_level > 0: #quality level; when 0 use all retrievals, with 1 only use recommended retrievals
                                qual_flags = unpackbits(data_SMAP_pm.retrieval_qual_flag_pm.values.astype(int),4)
                                mask_data_valid = mask_data_valid & (qual_flags[:,:,0]==0)
                            # 5) Select valid points: valid data, and within the given TSMP domain
                            sm = data_SMAP_pm.soil_moisture_pm.where(mask_data_valid).values[mask_data_valid]
                            lons = data_SMAP_pm.longitude_pm.where(mask_data_valid).values[mask_data_valid]
                            lats = data_SMAP_pm.latitude_pm.where(mask_data_valid).values[mask_data_valid]                            
                        else:
                            raise RuntimeError('mode should be am or pm')

                        # poor man's check that measurement points are on land and lie within the TSMP domain... There might be a better solution for this (convex hull check on EU-11 polygon?) 
                        mask_SMAP_lsm = griddata((np.concatenate((self.grid_TSMP.lon_centre.values.ravel(),self.grid_TSMP.lon_edges)),np.concatenate((self.grid_TSMP.lat_centre.values.ravel(),self.grid_TSMP.lat_edges))),
                                                 np.concatenate((self.grid_TSMP.lsm.values.ravel(),self.grid_TSMP.lsm_edges)),
                                                 (lons,lats),method='nearest') == 2

                        self.lons_out[date_] = lons[mask_SMAP_lsm]
                        self.lats_out[date_] = lats[mask_SMAP_lsm]
                        self.sm_out[date_] = sm[mask_SMAP_lsm]
        return self.flatten_y(self.sm_out)
                
    def interpolate_model_results(self,i_real,settings_run,indices_z=None,var='H2OSOI',history_stream='h0'):
        self.data_TSMP_i = {}
        self.files_clm = {}
        files_clm = sorted(glob(os.path.join(settings_run['dir_iter'],'R%3.3i/**/*.clm2.%s.*.nc'%(i_real,history_stream) )))
        # assert len(files_clm) == len(self.sm_out.keys()), 'Something might have gone wrong in realization %i: not every date has a matching file'%i_real

        for i1,date_ in enumerate(self.sm_out.keys()):
            date_find = str(date_.date())
            file_matching = [s for s in files_clm if date_find in s]
            
            if len(file_matching) > 1:
                print('%i matching files found (%s), taking the last file' %(len(file_matching),sorted(file_matching)))
                file_clm = sorted(file_matching)[-1]
            elif len(file_matching) == 1:
                file_clm = file_matching[0]
            else:
                file_clm = None
                raise RuntimeError('This operator only works for (more than) one matching model file per date, date:%s, i_real: %i, files: %s'%(date_find,i_real,file_matching) )
            
            self.files_clm[date_] = file_clm
            
            # file_clm = sorted(glob(os.path.join(settings_run['dir_iter'],'R%3.3i/**/clm.clm2.h0.*.nc'%i_real)))[i1]
            # print(date_,file_clm)
            
            data_TSMP = xr.open_dataset(file_clm)
            # add curvilinear lon/lat 
            data_TSMP = data_TSMP.assign_coords(lon_c=(('lat','lon'), self.grid_TSMP.lon_centre.values))
            data_TSMP = data_TSMP.assign_coords(lat_c=(('lat','lon'), self.grid_TSMP.lat_centre.values))

            # self.data_TSMP = data_TSMP
            if var == 'H2OSOI':
                SM = data_TSMP.H2OSOI[0]
            elif var == 'SOILLIQ': #base moisture on liquid part only
                eCLM_dz = np.array([0.02,0.04,0.06,0.08,0.120,
                                    0.160,0.200,0.240,0.280,0.320,
                                    0.360,0.400,0.440,0.540,0.640,
                                    0.740,0.840,0.940,1.040,1.140,
                                    2.390,4.676,7.635,11.140,15.115])
                SM = data_TSMP.SOILLIQ[0]/(eCLM_dz[0:20,np.newaxis,np.newaxis]*1000)
            else:
                raise RuntimeError('For soil moisture only H2OSOI and SOILLIQ supported')
                            
            
            if indices_z is None:
                data_TSMP_sm = SM[0].values
            elif type(indices_z) == int:
                data_TSMP_sm = SM[indices_z].values
            elif type(indices_z) == list:
                indices_select = np.ix_(np.array(indices_z),
                                        np.arange(SM.values.shape[1]),
                                        np.arange(SM.values.shape[2]))
                data_TSMP_sm = SM.values[indices_select].mean(axis=0)
            else:
                raise RuntimeError('indices_z should be None, int, or list of ints')
                
            mask_nan = ~np.isnan(data_TSMP_sm)
            
            self.data_TSMP_i[date_] = griddata((data_TSMP.lon_c.values[mask_nan],data_TSMP.lat_c.values[mask_nan]),data_TSMP_sm[mask_nan],(self.lons_out[date_],self.lats_out[date_]), method='nearest')

        # save most likely parameter output for plotting
        if i_real == 0:
            self.data_TSMP_ml = self.data_TSMP_i.copy()
            
        return self.flatten_y(self.data_TSMP_i)

    def plot_results(self,i_iter,i_real,settings_run,indices_z=None,var='H2OSOI',n_plots=None,dir_figs=None):

        print('Creating plots for iter %i, real. %i'%(i_iter,i_real))
        if dir_figs is None:
            dir_figs = os.path.join(settings_run['dir_iter'],'R%3.3i/figures'%i_real)
        if not os.path.exists(dir_figs):
            print('Creating folder to store DA information: %s' % (dir_figs) )
            os.mkdir(dir_figs)
            
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_extremes(under='k', over='r')

        if type(indices_z) == int:
            layer_plot = indices_z
        else:
            layer_plot = 0
            
        if n_plots is None:
            plot_every = 1
        elif n_plots >= len(list(self.data_TSMP_i.keys())):
            plot_every = 1
        else:
            plot_every = int(len(list(self.data_TSMP_i.keys())) // n_plots)
        for date_ in list(self.data_TSMP_i.keys())[::plot_every]:

            if len(self.data_TSMP_i[date_])>0:

                str_date = str(date_.date())

                # correlation
                min_ = min(min(self.data_TSMP_i[date_]),min(self.sm_out[date_]))
                max_ = max(max(self.data_TSMP_i[date_]),max(self.sm_out[date_]))
                R = pearsonr(self.data_TSMP_i[date_],self.sm_out[date_])[0]
                plt.figure(figsize=(5,5))
                plt.plot(self.data_TSMP_i[date_],self.sm_out[date_],'o',alpha=.7,markersize=3)
                plt.plot([min_,max_],[min_,max_],'k:')
                plt.xlabel('Modelled soil moisture')
                plt.ylabel('SMAP soil moisture')
                plt.title('%s, R=%3.3f (mean param. values)' % (date_,R) )
                plt.savefig(os.path.join(dir_figs,'corr_%s_%3.3i_R%3.3i.png'%(str_date,i_iter,i_real) ) )

                # difference
                plt.figure()
                plt.pcolormesh(self.grid_TSMP['lon_corner'],self.grid_TSMP['lat_corner'],self.grid_TSMP['lsm']==2,cmap=plt.cm.Greys,vmax=2 )
                diff = self.sm_out[date_] - self.data_TSMP_i[date_]
                # diff_max = max(np.abs(diff))
                diff_max = 0.4
                plt.scatter(self.lons_out[date_],self.lats_out[date_],s=10,c=diff,vmin=-diff_max,vmax=diff_max,cmap=plt.cm.bwr)
                cbar = plt.colorbar(extend='both')
                cbar.set_label('SMAP - TSMP [mm3/mm3]')
                plt.title('%s' % (date_) )
                plt.savefig(os.path.join(dir_figs,'mismatch_%s_%3.3i_R%3.3i.png'%(str_date,i_iter,i_real) ) )

                # TSMP results
                data_clm = xr.open_dataset(self.files_clm[date_])
                if var == 'H2OSOI':
                    SM = data_clm.H2OSOI[0]
                elif var == 'SOILLIQ': #base moisture on liquid part only
                    eCLM_dz = np.array([0.02,0.04,0.06,0.08,0.120,
                                        0.160,0.200,0.240,0.280,0.320,
                                        0.360,0.400,0.440,0.540,0.640,
                                        0.740,0.840,0.940,1.040,1.140,
                                        2.390,4.676,7.635,11.140,15.115])
                    SM = data_clm.SOILLIQ[0]/(eCLM_dz[0:20,np.newaxis,np.newaxis]*1000)
                else:
                    raise RuntimeError('For soil moisture only H2OSOI and SOILLIQ supported')

                plt.figure()
                plt.pcolormesh(self.grid_TSMP['lon_corner'],self.grid_TSMP['lat_corner'],SM[layer_plot],cmap=cmap,vmin=0.,vmax=1.)
                cbar = plt.colorbar(extend='max')
                cbar.set_label('CLM soil moisture [mm3/mm3]')
                plt.title('%s' % (date_) )
                plt.savefig(os.path.join(dir_figs,'TSMP_%s_%s_%3.3i_R%3.3i.png'%(var,str_date,i_iter,i_real) ) )

                # SMAP observations
                plt.figure()
                plt.pcolormesh(self.grid_TSMP['lon_corner'],self.grid_TSMP['lat_corner'],self.grid_TSMP['lsm']==2,cmap=plt.cm.Greys,vmax=2 )
                plt.scatter(self.lons_out[date_],self.lats_out[date_],s=10,c=self.sm_out[date_],vmin=0,vmax=1.,cmap=cmap)
                cbar = plt.colorbar(extend='max')
                cbar.set_label('SMAP soil moisture [mm3/mm3]')
                plt.title('%s' % (date_) )
                plt.savefig(os.path.join(dir_figs,'SMAP_%s_%3.3i_R%3.3i.png'%(str_date,i_iter,i_real) ) )

                
                plt.close('all')
                
                
class operator_clm_FLX:
    
    def __init__(self,file_lsm,file_corner,folder_FLX,ignore_rivers=False):
   
        # 1) get TSMP grid: is used to select SMAP results falling withing this domain. A corner file is required to discard points outside of the domain       
        self.grid_TSMP = get_TSMP_grid(file_lsm,file_corner,ignore_rivers=ignore_rivers)   
        self.folder_FLX = folder_FLX
        
    def flatten_y(self,y_in,var_=None):
        # flatten dict with values into a np.array
        if var_ == None:
            y_out = np.array([])
            for flx_name in y_in.keys():
                y_out = np.append(y_out,list(y_in[flx_name].values()))        
        else:
            y_out = np.array([])
            for flx_name in y_in.keys():
                y_out = np.append(y_out,list(y_in[flx_name][var_].values()))     
        return y_out
    
#     def get_measurements_HH(self,date_results_iter,date_DA_start=datetime(1900,1,1),date_DA_end=datetime(2200,1,1)):
#         ### Get hourly fluxnet measurements -> not used anymore, standard is daily
#         # outputdt in lnd_in needs to be set to hourly

#         self.data_flx = {}
#         self.file_FLX_stations = os.path.join(self.folder_FLX, 'stations.csv')
        
#         data_stations = pd.read_csv(self.file_FLX_stations)
#         flx_names = data_stations['FLX_name']

#         # loop through the fluxnet stations
#         for flx_name,flx_lon,flx_lat in zip(data_stations['FLX_name'], data_stations['lon'], data_stations['lat']):

#             self.data_flx[flx_name] = {}
#             self.data_flx[flx_name]['lon'] = flx_lon
#             self.data_flx[flx_name]['lat'] = flx_lat
#             self.data_flx[flx_name]['LE_CORR'] = {}
#             self.data_flx[flx_name]['LE_RANDUNC'] = {}

#             # then loop through the dates
#             yr_current = 0
#             valid_location = False
#             for date_results in date_results_iter:

#                 for date_ in date_results[1:]:

#                     # only load in the data if the year has changed (I separated the FLUXNET data per year to keep file size small)
#                     if date_.year != yr_current:
#                         yr_current = date_.year

#                         # Define the column containing the date
#                         date_column = 'TIMESTAMP_START'
#                         # Define a function to parse the date string to datetime
#                         date_parser = lambda x: pd.to_datetime(x, format='%Y%m%d%H%M')

#                         file = glob('/p/scratch/cjibg36/kaandorp2/data/FLUXNET/%i/FLX_%s_FLUXNET2015_FULLSET_HH_*.csv'%(yr_current,flx_name))[0]
#                         flx_data = pd.read_csv(file,parse_dates=[date_column])

#                         # check if the lon/lat fall within the domain
#                         valid_location = griddata((np.concatenate((self.grid_TSMP.lon_centre.values.ravel(),self.grid_TSMP.lon_edges)),
#                                                    np.concatenate((self.grid_TSMP.lat_centre.values.ravel(),self.grid_TSMP.lat_edges))),
#                                                  np.concatenate((self.grid_TSMP.lsm.values.ravel(),self.grid_TSMP.lsm_edges)),
#                                                  (flx_lon,flx_lat),method='nearest') == 2
#                         if len(flx_data) == 0:
#                             valid_location = False
                            
#                         if not valid_location:
#                             print('%s falling outside of TSMP domain (time and/or space)' % flx_name)


#                     if date_ < date_DA_start or date_ > date_DA_end:
#                         # print('Skipping %s: outside of range %s-%s'%(str(date_),str(date_DA_start),str(date_DA_end)) )
#                         pass
#                     elif valid_location:
#                         #for the given model date (date_) get the corresponding FLUXNET data
#                         try:
#                             i_closest = np.argmin(np.abs(flx_data['TIMESTAMP_START'] - date_))
#                             diff_days = abs(flx_data['TIMESTAMP_START'][i_closest] - date_)
#                         except:
#                             print(file)
#                             print(flx_data['TIMESTAMP_START'], date_)
#                         if diff_days > timedelta(hours=1):
#                             pass
#                             # print('Measurement and model output differ more than a hour, FLUXNET: %s, TSMP: %s'%(flx_data['TIMESTAMP_START'][i_closest],date_))
#                             # This happens e.g. when not every day has a measurement
#                         else:
#                             flx_data_ = flx_data.iloc[i_closest,:]

#                             if flx_data_.LE_CORR == -9999:
#                                 pass #skip if values are 'nan'
#                             else:
#                                 self.data_flx[flx_name]['LE_CORR'][date_] = flx_data_.LE_CORR / 2.45e6
#                                 self.data_flx[flx_name]['LE_RANDUNC'][date_] = flx_data_.LE_RANDUNC / 2.45e6

#             if len(self.data_flx[flx_name]['LE_CORR'].keys()) == 0:
#                 print('No corrected LE values found for %s' % flx_name)
#                 del self.data_flx[flx_name]

#         # return self.flatten_y(self.sm_out)
#         return self.data_flx        
 

    def get_measurements(self,date_results_iter,date_DA_start=datetime(1900,1,1),date_DA_end=datetime(2200,1,1)):
        ### Get daily averaged fluxnet measurements 
        # outputdt in lnd_in needs to be set to daily

        self.data_flx = {}
        self.file_FLX_stations = os.path.join(self.folder_FLX, 'stations.csv')
        
        data_stations = pd.read_csv(self.file_FLX_stations)
        flx_names = data_stations['FLX_name']

        # loop through the fluxnet stations
        for flx_name,flx_lon,flx_lat in zip(data_stations['FLX_name'], data_stations['lon'], data_stations['lat']):
            
            # flag, set to True when the location falls within the domain
            valid_location = False

            self.data_flx[flx_name] = {}
            self.data_flx[flx_name]['lon'] = flx_lon
            self.data_flx[flx_name]['lat'] = flx_lat
            self.data_flx[flx_name]['LE_CORR'] = {}
            self.data_flx[flx_name]['LE_RANDUNC'] = {}

            # load in the data, check if location is valid
            date_parser = lambda x: pd.to_datetime(x, format='%Y%m%d')
            date_column = 'TIMESTAMP'
            file = glob('/p/scratch/cjibg36/kaandorp2/data/FLUXNET/FLX_%s_FLUXNET2015_FULLSET_DD_*.csv'%(flx_name))[0]
            flx_data = pd.read_csv(file,parse_dates=[date_column])

            # check if the lon/lat fall within the domain
            valid_location = griddata((np.concatenate((self.grid_TSMP.lon_centre.values.ravel(),self.grid_TSMP.lon_edges)),
                                       np.concatenate((self.grid_TSMP.lat_centre.values.ravel(),self.grid_TSMP.lat_edges))),
                                     np.concatenate((self.grid_TSMP.lsm.values.ravel(),self.grid_TSMP.lsm_edges)),
                                     (flx_lon,flx_lat),method='nearest') == 2
            if len(flx_data) == 0:
                valid_location = False

            if not valid_location:
                print('%s falling outside of TSMP domain (time and/or space)' % flx_name)

            
            # then loop through the dates
            diff_days = timedelta(days=9999)
            for date_results in date_results_iter:

                for date_ in date_results[1:]:

                    date_ = pd.Timestamp(date_.date()) #compare daily averages: remove hours from timestamp
                    
                    if date_ < date_DA_start or date_ > date_DA_end:
                        # print('Skipping %s: outside of range %s-%s'%(str(date_),str(date_DA_start),str(date_DA_end)) )
                        pass
                    elif valid_location:
                        #for the given model date (date_) get the corresponding FLUXNET data
                        try:
                            i_closest = np.argmin(np.abs(flx_data['TIMESTAMP'] - date_))
                            diff_days = abs(flx_data['TIMESTAMP'][i_closest] - date_)
                        except:
                            print(file)
                            print(flx_data['TIMESTAMP'], date_)
                        if diff_days > timedelta(days=1):
                            pass
                            # print('Measurement and model output differ more than a hour, FLUXNET: %s, TSMP: %s'%(flx_data['TIMESTAMP'][i_closest],date_))
                            # This happens e.g. when not every day has a measurement
                        else:
                            flx_data_ = flx_data.iloc[i_closest,:]

                            if flx_data_.LE_CORR == -9999 or flx_data_.LE_RANDUNC < 0:
                                pass #skip if values are 'nan', or if uncertainty is not defined (negative values)
                            else:
                                self.data_flx[flx_name]['LE_CORR'][date_] = flx_data_.LE_CORR * 0.035
                                self.data_flx[flx_name]['LE_RANDUNC'][date_] = flx_data_.LE_RANDUNC * 0.035

            if len(self.data_flx[flx_name]['LE_CORR'].keys()) == 0:
                print('No corrected LE values found for %s' % flx_name)
                del self.data_flx[flx_name]

        # return self.flatten_y(self.sm_out)
        return self.data_flx        
    
    
    def interpolate_model_results(self,i_real,settings_run,var='QFLX_EVAP_TOT',history_stream='h1'):
        self.data_TSMP_i = {}
        self.files_clm = {}
        files_clm = sorted(glob(os.path.join(settings_run['dir_iter'],'R%3.3i/**/*.clm2.%s.*.nc'%(i_real,history_stream) )))
        # assert len(files_clm) == len(self.sm_out.keys()), 'Something might have gone wrong in realization %i: not every date has a matching file'%i_real

        for flx_name in self.data_flx.keys():

            lon_ = self.data_flx[flx_name]['lon']
            lat_ = self.data_flx[flx_name]['lat']
            
            self.data_TSMP_i[flx_name] = {}
            
            for i1,date_ in enumerate(self.data_flx[flx_name]['LE_CORR'].keys()):

                date_find = str(date_.date())
                file_matching = [s for s in files_clm if date_find in s]
                
                if len(file_matching) > 1:
                    print('%i matching files found (%s), taking the last file' %(len(file_matching),sorted(file_matching)))
                    file_clm = sorted(file_matching)[-1]
                elif len(file_matching) == 1:
                    file_clm = file_matching[0]
                else:
                    file_clm = None
                    raise RuntimeError('This operator only works for (more than) one matching model file per date, date:%s, i_real: %i, files: %s'%(date_find,i_real,file_matching) )

                self.files_clm[date_] = file_clm

                data_TSMP = xr.open_dataset(file_clm)
                # add curvilinear lon/lat 
                data_TSMP = data_TSMP.assign_coords(lon_c=(('lat','lon'), self.grid_TSMP.lon_centre.values))
                data_TSMP = data_TSMP.assign_coords(lat_c=(('lat','lon'), self.grid_TSMP.lat_centre.values))

                LE = data_TSMP[var][0].values * (24*60*60) # s-1 to d-1

                mask_nan = ~np.isnan(LE)

                self.data_TSMP_i[flx_name][date_] = griddata((data_TSMP.lon_c.values[mask_nan],data_TSMP.lat_c.values[mask_nan]),
                                                             LE[mask_nan],
                                                             (lon_,lat_), method='nearest')
                
    def plot_results(self,i_iter,i_real,settings_run,dir_figs=None):

        print('Creating plots for iter %i, real. %i'%(i_iter,i_real))
        if dir_figs is None:
            dir_figs = os.path.join(settings_run['dir_iter'],'R%3.3i/figures'%i_real)
        if not os.path.exists(dir_figs):
            print('Creating folder to store FLX information: %s' % (dir_figs) )
            os.mkdir(dir_figs)
          
        # 1) time series per location
        for flx_name in list(self.data_TSMP_i.keys()):
            plt.figure(figsize=(12,8))
            plt.errorbar(self.data_flx[flx_name]['LE_CORR'].keys(),self.data_flx[flx_name]['LE_CORR'].values(),
                         list(self.data_flx[flx_name]['LE_RANDUNC'].values()),fmt='o',capsize=2)
            plt.plot(list(self.data_TSMP_i[flx_name].keys()),list(self.data_TSMP_i[flx_name].values()),'ko')
            R = pearsonr(list(self.data_flx[flx_name]['LE_CORR'].values()),list(self.data_TSMP_i[flx_name].values()))[0]
            rmse = np.sqrt(np.mean((np.array(list(self.data_flx[flx_name]['LE_CORR'].values()))-np.array(list(self.data_TSMP_i[flx_name].values())))**2))
            
            plt.xlabel('Date')
            plt.ylabel('ET [mm/d]')
            plt.title('%s, R=%3.3f, RMSE=%3.2e' % (flx_name,R,rmse))
            plt.savefig(os.path.join(dir_figs,'FLX_timeseries_%s_%3.3i_R%3.3i.png'%(flx_name,i_iter,i_real) ) )

            plt.close('all')
            
        # 2) correlation of all measurements together
        y_all = self.flatten_y(self.data_flx,'LE_CORR')
        x_all = self.flatten_y(self.data_TSMP_i)
        min_ = min(min(x_all),min(y_all))
        max_ = max(max(x_all),max(y_all))

        R = pearsonr(x_all,y_all)[0]
        rmse = np.sqrt(np.mean((x_all-y_all)**2))
        
        plt.figure(figsize=(5,5))
        plt.plot(x_all,y_all,'o',alpha=.7,markersize=3)
        plt.plot([min_,max_],[min_,max_],'k:')
        plt.xlabel('Modelled ET [mm/d]')
        plt.ylabel('FLUXNET ET [mm/d]')
        plt.title('Correlation all FLUXNET stations, \nR=%3.3f, RMSE=%3.2e' % (R,rmse))
        plt.savefig(os.path.join(dir_figs,'FLX_corr_all_%3.3i_R%3.3i.png'%(i_iter,i_real) ) )
        
        
        # 3) helper plot of all locations     
        symbols = ['x','o','d','+','^','v']
        plt.figure(figsize=(12,9))
        plt.pcolormesh(self.grid_TSMP['lon_corner'],self.grid_TSMP['lat_corner'],self.grid_TSMP['lsm']==2,cmap=plt.cm.Greys,vmax=2 )
        for i1,flx_name in enumerate(self.data_flx.keys()):
            i_color = i1 % 10
            i_symbol = i1 // 10

            plt.plot(self.data_flx[flx_name]['lon'],self.data_flx[flx_name]['lat'],symbols[i_symbol],color=plt.cm.tab10(i_color),label=flx_name)
        plt.legend(fontsize=8,ncol=3)
        plt.savefig(os.path.join(dir_figs,'FLX_locations.png' ) )
        
        plt.close('all')