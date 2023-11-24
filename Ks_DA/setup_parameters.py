import numpy as np
import sys
extra_path = '/p/project/cjibg36/kaandorp2/Git/SLOTH' 
if extra_path not in sys.path:
    sys.path.append(extra_path)
from sloth.IO import readSa, writeSa
import os
import xarray as xr


def setup_Ks(settings_setup,settings_run):

    file_indi = settings_setup['file_indi']
    sample_every_xy = settings_setup['Ks_sample_xy']
    sample_every_z = settings_setup['Ks_sample_z']
    dir_out = settings_run['dir_DA']
    
    ### indicator file data

    data_indi = readSa(file_indi)

    indi_names = ['TC%2.2i' %i for i in range(1,13)] + ['Allv'] + ['BGR%1.1i' %i for i in range(1,7)] + ['Lake','Sea']
    indi_nr = np.setdiff1d(np.arange(1,len(indi_names)+2),np.array([20]) ) #array starting at 1, to 22 and without 20
    indi_nr_estim = np.setdiff1d(np.arange(1,len(indi_names)+2),np.array([13,20,21,22]) ) #which indicators to estimate: don't estimate Allv, water
    indi_perm = [0.267787,0.043832,0.015951,0.005021,0.0076,0.01823,0.005493,0.00341,0.004632,0.004729,0.004007,0.006149,
                 0.1,
                 0.1,0.05,0.01,0.005,0.001,0.0005,
                 1e-5,1e-5]
    indi_logKs     = [2.808,2.022,1.583,1.081,1.261,1.641,
                      1.120,0.913,1.046,1.055,0.983,1.169]
    indi_logKs_sig = [0.590,0.640,0.660,0.920,0.740,0.270,
                      0.850,1.090,0.760,0.890,0.570,0.920]
    indi_names_full = ['sand','loamy sand','sandy loam','loam','silt loam','silt',
                       'sandy clay loam','clay loam','silty clay loam','sandy clay','silty clay','clay'] + ['Alluvium'] + ['BGR%1.1i' %i for i in range(1,7)] + ['Lake','Sea']
    indi_dz = 2*np.array([9.0,7.5,5.0,5.0,2.0,0.5,0.35,0.25,0.15,0.10,0.065,0.035,0.025,0.015,0.01])
    indi_depths = np.flip(np.cumsum(np.flip(indi_dz)))


    ### setting sampling and interpolation indices

    # select z-indices for which the Kriged fields will be created. For the rest of the indices, interpolate the Ks values
    i_sample_z = np.arange(data_indi.shape[0]-1,0,-sample_every_z)
    if 0 not in i_sample_z:
        i_sample_z = np.append(i_sample_z,0)
    i_sample_z = np.flip(i_sample_z)
    assert i_sample_z[0] == 0
    assert i_sample_z[-1] == data_indi.shape[0]-1

    # indices of layers for which linear interpolation is used
    i_interp_z = np.setxor1d(i_sample_z,np.arange(0,data_indi.shape[0]))

    ### generate Ks field from indicator data
    field_Ks_log = np.zeros(data_indi.shape)
    field_Ks_sig_log = np.zeros(data_indi.shape)
    sig_log_standard = 0.74 #set standard uncertainty to mean of Rosetta values

    for i1,indicator in enumerate(indi_nr):
        mask = (data_indi == indicator)

        if indicator <= 12: #surface layers, soilgrids
            # use Rosetta: values are given in log10(cm/day) -> convert to log10(m/h) for ParFlow
            field_Ks_log[mask] = indi_logKs[i1] - np.log10(100*24) #go from cm/day (Rosetta) to m/h (ParFlow)
            field_Ks_sig_log[mask] = indi_logKs_sig[i1] # no unit transform necessary here (log space is already relative)
        elif indicator < 20: # deep subsurface
            # use standard ParFlow values. Take the log10, and use predefined uncertainty
            field_Ks_log[mask] = np.log10(indi_perm[i1])
            field_Ks_sig_log[mask] = sig_log_standard
        else: #lake and ocean
            field_Ks_log[mask] = np.log10(indi_perm[i1])
            field_Ks_sig_log[mask] = np.nan

    assert (field_Ks_log==0).sum() == 0

    ### sample the Ks field at given locations
    i_c_z = i_sample_z
    i_c_lat = np.arange(0,field_Ks_log.shape[1],sample_every_xy)
    i_c_lon = np.arange(0,field_Ks_log.shape[2],sample_every_xy)

    X_c_all = np.meshgrid(i_c_z,i_c_lat,i_c_lon,indexing='ij')
    # mask_c_land = ~np.isnan(field_Ks_sig_log[np.ix_(i_c_z,i_c_lat,i_c_lon)])
    mask_c_land = np.isin(data_indi,indi_nr_estim)[np.ix_(i_c_z,i_c_lat,i_c_lon)]
    X_train = np.array([X_c_all[0][mask_c_land],
                       X_c_all[1][mask_c_land],
                       X_c_all[2][mask_c_land]]).T

    Y_train = field_Ks_log[np.ix_(i_c_z,i_c_lat,i_c_lon)][mask_c_land]
    Y_sigma = field_Ks_sig_log[np.ix_(i_c_z,i_c_lat,i_c_lon)][mask_c_land]


    ### write static data: x-vals, y-vals, z-vals
    np.save(os.path.join(dir_out,'Ks.static'),X_train)

    ### write parameter data: log(Ks) log(sigma)
    np.save(os.path.join(dir_out,'Ks.param.000.000.prior'),np.array([Y_train,Y_sigma]).T)

    
def setup_Ks_anom(settings_setup,settings_run):
    
    ## Get indicator data ##
    file_indi = settings_setup['file_indi']
    sample_every_xy = settings_setup['Ks_sample_xy']
    sample_every_z = settings_setup['Ks_sample_z']
    dir_out = settings_run['dir_DA']
    
    data_indi = readSa(file_indi)

    indi_names = ['TC%2.2i' %i for i in range(1,13)] + ['Allv'] + ['BGR%1.1i' %i for i in range(1,7)] + ['Lake','Sea']
    indi_nr = np.setdiff1d(np.arange(1,len(indi_names)+2),np.array([20]) ) #array starting at 1, to 22 and without 20
    indi_nr_estim = np.setdiff1d(np.arange(1,len(indi_names)+2),np.array([13,20,21,22]) ) #which indicators to estimate: don't estimate Allv, water
    indi_perm = [0.267787,0.043832,0.015951,0.005021,0.0076,0.01823,0.005493,0.00341,0.004632,0.004729,0.004007,0.006149,
                 0.1,
                 0.1,0.05,0.01,0.005,0.001,0.0005,
                 1e-5,1e-5]
    indi_logKs     = [2.808,2.022,1.583,1.081,1.261,1.641,
                      1.120,0.913,1.046,1.055,0.983,1.169]
    indi_logKs_sig = [0.590,0.640,0.660,0.920,0.740,0.270,
                      0.850,1.090,0.760,0.890,0.570,0.920]
    indi_names_full = ['sand','loamy sand','sandy loam','loam','silt loam','silt',
                       'sandy clay loam','clay loam','silty clay loam','sandy clay','silty clay','clay'] + ['Alluvium'] + ['BGR%1.1i' %i for i in range(1,7)] + ['Lake','Sea']
    indi_dz = 2*np.array([9.0,7.5,5.0,5.0,2.0,0.5,0.35,0.25,0.15,0.10,0.065,0.035,0.025,0.015,0.01])
    indi_depths = np.flip(np.cumsum(np.flip(indi_dz)))
    
    
    ### setting sampling and interpolation indices

    # select z-indices for which the Kriged fields will be created. For the rest of the indices, interpolate the Ks values
    i_sample_z = np.arange(data_indi.shape[0]-1,0,-sample_every_z)
    if 0 not in i_sample_z:
        i_sample_z = np.append(i_sample_z,0)
    i_sample_z = np.flip(i_sample_z)
    assert i_sample_z[0] == 0
    assert i_sample_z[-1] == data_indi.shape[0]-1

    ### generate anomaly field from indicator data
    field_anom_sig_log = np.zeros(data_indi.shape)
    sig_log_standard = 0.74 #set standard uncertainty to mean of Rosetta values

    for i1,indicator in enumerate(indi_nr):
        mask = (data_indi == indicator)

        if indicator <= 12: #surface layers, soilgrids
            field_anom_sig_log[mask] = indi_logKs_sig[i1] # no unit transform necessary here (log space is already relative)
        elif indicator < 20: # deep subsurface
            field_anom_sig_log[mask] = sig_log_standard
        else: #lake and ocean
            field_anom_sig_log[mask] = np.nan

    assert (field_anom_sig_log==0).sum() == 0


    ### sample the Ks field at given locations
    i_c_z = i_sample_z
    i_c_lat = np.arange(0,field_anom_sig_log.shape[1],sample_every_xy)
    i_c_lon = np.arange(0,field_anom_sig_log.shape[2],sample_every_xy)

    X_c_all = np.meshgrid(i_c_z,i_c_lat,i_c_lon,indexing='ij')
    mask_c_land = np.isin(data_indi,indi_nr_estim)[np.ix_(i_c_z,i_c_lat,i_c_lon)]
    X_train = np.array([X_c_all[0][mask_c_land],
                       X_c_all[1][mask_c_land],
                       X_c_all[2][mask_c_land]]).T

    Y_sigma = field_anom_sig_log[np.ix_(i_c_z,i_c_lat,i_c_lon)][mask_c_land]
    Y_train = np.zeros(Y_sigma.shape)

    ### write static data: x-vals, y-vals, z-vals
    np.save(os.path.join(dir_out,'Ks_anom.static'),X_train)

    ### write parameter data: log(Ks) log(sigma)
    np.save(os.path.join(dir_out,'Ks_anom.param.000.000.prior'),np.array([Y_train,Y_sigma]).T)
    
    
def setup_a_anom(settings_setup,settings_run):
    
    ## Get indicator data ##
    file_indi = settings_setup['file_indi']
    sample_every_xy = settings_setup['a_sample_xy']
    sample_every_z = settings_setup['a_sample_z']
    dir_out = settings_run['dir_DA']
    
    data_indi = readSa(file_indi)

    indi_names = ['TC%2.2i' %i for i in range(1,13)] + ['Allv'] + ['BGR%1.1i' %i for i in range(1,7)] + ['Lake','Sea']
    indi_nr = np.setdiff1d(np.arange(1,len(indi_names)+2),np.array([20]) ) #array starting at 1, to 22 and without 20
    indi_nr_estim = np.setdiff1d(np.arange(1,len(indi_names)+2),np.array([13,20,21,22]) ) #which indicators to estimate: don't estimate Allv, water
    indi_loga = [np.log10(2.0)]*len(indi_nr) # standard alpha values, in m-1
    Ros_loga     = [-1.453,-1.459,-1.574,-1.954,-2.296,-2.182,
                    -1.676,-1.801,-2.076,-1.476,-1.790,-1.825] # Rosetta values for alpha, in cm-1
    Ros_loga_sig = [0.25,0.47,0.56,0.73,0.57,0.30,
                    0.71,0.69,0.59,0.57,0.64,0.68] # Rosetta uncertainty
    indi_names_full = ['sand','loamy sand','sandy loam','loam','silt loam','silt',
                       'sandy clay loam','clay loam','silty clay loam','sandy clay','silty clay','clay'] + ['Alluvium'] + ['BGR%1.1i' %i for i in range(1,7)] + ['Lake','Sea']
    indi_dz = 2*np.array([9.0,7.5,5.0,5.0,2.0,0.5,0.35,0.25,0.15,0.10,0.065,0.035,0.025,0.015,0.01])
    indi_depths = np.flip(np.cumsum(np.flip(indi_dz)))   
    
    
    ### setting sampling and interpolation indices

    # select z-indices for which the Kriged fields will be created. For the rest of the indices, interpolate the alpha values
    i_sample_z = np.arange(data_indi.shape[0]-1,0,-sample_every_z)
    if 0 not in i_sample_z:
        i_sample_z = np.append(i_sample_z,0)
    i_sample_z = np.flip(i_sample_z)
    assert i_sample_z[0] == 0
    assert i_sample_z[-1] == data_indi.shape[0]-1

    ### generate anomaly field from indicator data
    field_anom_sig_log = np.zeros(data_indi.shape)
    sig_log_standard = 0.56 #set standard uncertainty to mean of Rosetta values

    for i1,indicator in enumerate(indi_nr):
        mask = (data_indi == indicator)

        if indicator <= 12: #surface layers, soilgrids
            field_anom_sig_log[mask] = Ros_loga_sig[i1] # no unit transform necessary here (log space is already relative)
        elif indicator < 20: # deep subsurface
            field_anom_sig_log[mask] = sig_log_standard
        else: #lake and ocean
            field_anom_sig_log[mask] = np.nan

    assert (field_anom_sig_log==0).sum() == 0


    ### sample the alpha field at given locations
    i_c_z = i_sample_z
    i_c_lat = np.arange(0,field_anom_sig_log.shape[1],sample_every_xy)
    i_c_lon = np.arange(0,field_anom_sig_log.shape[2],sample_every_xy)

    X_c_all = np.meshgrid(i_c_z,i_c_lat,i_c_lon,indexing='ij')
    mask_c_land = np.isin(data_indi,indi_nr_estim)[np.ix_(i_c_z,i_c_lat,i_c_lon)]
    X_train = np.array([X_c_all[0][mask_c_land],
                       X_c_all[1][mask_c_land],
                       X_c_all[2][mask_c_land]]).T

    Y_sigma = field_anom_sig_log[np.ix_(i_c_z,i_c_lat,i_c_lon)][mask_c_land]
    Y_train = np.zeros(Y_sigma.shape)

    ### write static data: x-vals, y-vals, z-vals
    np.save(os.path.join(dir_out,'a_anom.static'),X_train)

    ### write parameter data: anomaly of alpha (0) and std
    np.save(os.path.join(dir_out,'a_anom.param.000.000.prior'),np.array([Y_train,Y_sigma]).T)
    

def setup_n_anom(settings_setup,settings_run):
    
    ## Get indicator data ##
    file_indi = settings_setup['file_indi']
    sample_every_xy = settings_setup['n_sample_xy']
    sample_every_z = settings_setup['n_sample_z']
    dir_out = settings_run['dir_DA']
    
    data_indi = readSa(file_indi)

    indi_names = ['TC%2.2i' %i for i in range(1,13)] + ['Allv'] + ['BGR%1.1i' %i for i in range(1,7)] + ['Lake','Sea']
    indi_nr = np.setdiff1d(np.arange(1,len(indi_names)+2),np.array([20]) ) #array starting at 1, to 22 and without 20
    indi_nr_estim = np.setdiff1d(np.arange(1,len(indi_names)+2),np.array([13,20,21,22]) ) #which indicators to estimate: don't estimate Allv, water
    indi_logn = [np.log10(4.0)]*len(indi_nr) # standard n values
    Ros_logn     = [0.502,0.242,0.161,0.168,0.221,0.225,
                    0.124,0.151,0.182,0.082,0.121,0.098] # Rosetta values for alpha
    Ros_logn_sig = [0.18,0.16,0.11,0.13,0.14,0.13,
                    0.12,0.12,0.13,0.06,0.10,0.07] # Rosetta uncertainty
    indi_names_full = ['sand','loamy sand','sandy loam','loam','silt loam','silt',
                       'sandy clay loam','clay loam','silty clay loam','sandy clay','silty clay','clay'] + ['Alluvium'] + ['BGR%1.1i' %i for i in range(1,7)] + ['Lake','Sea']
    indi_dz = 2*np.array([9.0,7.5,5.0,5.0,2.0,0.5,0.35,0.25,0.15,0.10,0.065,0.035,0.025,0.015,0.01])
    indi_depths = np.flip(np.cumsum(np.flip(indi_dz)))
    
    
    ### setting sampling and interpolation indices

    # select z-indices for which the Kriged fields will be created. For the rest of the indices, interpolate the alpha values
    i_sample_z = np.arange(data_indi.shape[0]-1,0,-sample_every_z)
    if 0 not in i_sample_z:
        i_sample_z = np.append(i_sample_z,0)
    i_sample_z = np.flip(i_sample_z)
    assert i_sample_z[0] == 0
    assert i_sample_z[-1] == data_indi.shape[0]-1

    ### generate anomaly field from indicator data
    field_anom_sig_log = np.zeros(data_indi.shape)
    sig_log_standard = 0.12 #set standard uncertainty to mean of Rosetta values

    for i1,indicator in enumerate(indi_nr):
        mask = (data_indi == indicator)

        if indicator <= 12: #surface layers, soilgrids
            field_anom_sig_log[mask] = Ros_logn_sig[i1] # no unit transform necessary here (log space is already relative)
        elif indicator < 20: # deep subsurface
            field_anom_sig_log[mask] = sig_log_standard
        else: #lake and ocean
            field_anom_sig_log[mask] = np.nan

    assert (field_anom_sig_log==0).sum() == 0


    ### sample the alpha field at given locations
    i_c_z = i_sample_z
    i_c_lat = np.arange(0,field_anom_sig_log.shape[1],sample_every_xy)
    i_c_lon = np.arange(0,field_anom_sig_log.shape[2],sample_every_xy)

    X_c_all = np.meshgrid(i_c_z,i_c_lat,i_c_lon,indexing='ij')
    mask_c_land = np.isin(data_indi,indi_nr_estim)[np.ix_(i_c_z,i_c_lat,i_c_lon)]
    X_train = np.array([X_c_all[0][mask_c_land],
                       X_c_all[1][mask_c_land],
                       X_c_all[2][mask_c_land]]).T

    Y_sigma = field_anom_sig_log[np.ix_(i_c_z,i_c_lat,i_c_lon)][mask_c_land]
    Y_train = np.zeros(Y_sigma.shape)

    ### write static data: x-vals, y-vals, z-vals
    np.save(os.path.join(dir_out,'n_anom.static'),X_train)

    ### write parameter data: anomaly of alpha (0) and std
    np.save(os.path.join(dir_out,'n_anom.param.000.000.prior'),np.array([Y_train,Y_sigma]).T)
    
    
def setup_poros_anom(settings_setup,settings_run):
    
    ## Get indicator data ##
    file_indi = settings_setup['file_indi']
    sample_every_xy = settings_setup['poros_sample_xy']
    sample_every_z = settings_setup['poros_sample_z']
    dir_out = settings_run['dir_DA']
    
    data_indi = readSa(file_indi)

    indi_names = ['TC%2.2i' %i for i in range(1,13)] + ['Allv'] + ['BGR%1.1i' %i for i in range(1,7)] + ['Lake','Sea']
    indi_nr = np.setdiff1d(np.arange(1,len(indi_names)+2),np.array([20]) ) #array starting at 1, to 22 and without 20
    indi_nr_estim = np.setdiff1d(np.arange(1,len(indi_names)+2),np.array([13,20,21,22]) ) #which indicators to estimate: don't estimate Allv, water
    indi_poros = [0.4]*len(indi_nr) # standard porosity values
    Ros_poros     = [0.375,0.390,0.387,0.399,0.439,0.489,
                    0.384,0.442,0.482,0.385,0.481,0.459] # Rosetta values for porosity
    Ros_poros_sig = [0.055,0.070,0.085,0.098,0.093,0.078,
                    0.061,0.079,0.086,0.046,0.080,0.079] # Rosetta uncertainty
    indi_names_full = ['sand','loamy sand','sandy loam','loam','silt loam','silt',
                       'sandy clay loam','clay loam','silty clay loam','sandy clay','silty clay','clay'] + ['Alluvium'] + ['BGR%1.1i' %i for i in range(1,7)] + ['Lake','Sea']
    indi_dz = 2*np.array([9.0,7.5,5.0,5.0,2.0,0.5,0.35,0.25,0.15,0.10,0.065,0.035,0.025,0.015,0.01])
    indi_depths = np.flip(np.cumsum(np.flip(indi_dz)))
    
    
    ### setting sampling and interpolation indices

    # select z-indices for which the Kriged fields will be created. For the rest of the indices, interpolate the alpha values
    i_sample_z = np.arange(data_indi.shape[0]-1,0,-sample_every_z)
    if 0 not in i_sample_z:
        i_sample_z = np.append(i_sample_z,0)
    i_sample_z = np.flip(i_sample_z)
    assert i_sample_z[0] == 0
    assert i_sample_z[-1] == data_indi.shape[0]-1

    ### generate anomaly field from indicator data
    field_anom_sig_log = np.zeros(data_indi.shape)
    sig_log_standard = 0.076 #set standard uncertainty to mean of Rosetta values

    for i1,indicator in enumerate(indi_nr):
        mask = (data_indi == indicator)

        if indicator <= 12: #surface layers, soilgrids
            field_anom_sig_log[mask] = Ros_poros_sig[i1] # no unit transform necessary here (log space is already relative)
        elif indicator < 20: # deep subsurface
            field_anom_sig_log[mask] = sig_log_standard
        else: #lake and ocean
            field_anom_sig_log[mask] = np.nan

    assert (field_anom_sig_log==0).sum() == 0


    ### sample the alpha field at given locations
    i_c_z = i_sample_z
    i_c_lat = np.arange(0,field_anom_sig_log.shape[1],sample_every_xy)
    i_c_lon = np.arange(0,field_anom_sig_log.shape[2],sample_every_xy)

    X_c_all = np.meshgrid(i_c_z,i_c_lat,i_c_lon,indexing='ij')
    mask_c_land = np.isin(data_indi,indi_nr_estim)[np.ix_(i_c_z,i_c_lat,i_c_lon)]
    X_train = np.array([X_c_all[0][mask_c_land],
                       X_c_all[1][mask_c_land],
                       X_c_all[2][mask_c_land]]).T

    Y_sigma = field_anom_sig_log[np.ix_(i_c_z,i_c_lat,i_c_lon)][mask_c_land]
    Y_train = np.zeros(Y_sigma.shape)

    ### write static data: x-vals, y-vals, z-vals
    np.save(os.path.join(dir_out,'poros_anom.static'),X_train)

    ### write parameter data: anomaly of alpha (0) and std
    np.save(os.path.join(dir_out,'poros_anom.param.000.000.prior'),np.array([Y_train,Y_sigma]).T)
    

def setup_slope_anom(settings_setup,settings_run):
    
    ## Get slope data ##
    file_slopex = settings_setup['file_slopex']
    file_slopey = settings_setup['file_slopey']
    file_indi = settings_setup['file_indi']
    sample_every_xy = settings_setup['slope_sample_xy']
    dir_out = settings_run['dir_DA']
    slope_std = 0.5 # set 2 sigma to one order of magnitude
    
    data_x = readSa(file_slopex)
    data_indi = readSa(file_indi)

    field_anom_sig_log = slope_std*np.ones(data_indi.shape)
    
    indi_names = ['TC%2.2i' %i for i in range(1,13)] + ['Allv'] + ['BGR%1.1i' %i for i in range(1,7)] + ['Lake','Sea']
    indi_nr_estim = np.setdiff1d(np.arange(1,len(indi_names)+2),np.array([13,20,21,22]) ) #which indicators to estimate: don't estimate Allv, water

    ### setting sampling and interpolation indices
    # slope only has one layer -> sample 0th element
    i_sample_z = 0

    ### sample the alpha field at given locations
    i_c_z = np.array([i_sample_z])
    i_c_lat = np.arange(0,data_x.shape[1],sample_every_xy)
    i_c_lon = np.arange(0,data_x.shape[2],sample_every_xy)

    X_c_all = np.meshgrid(i_c_z,i_c_lat,i_c_lon,indexing='ij')
    mask_c_land = np.isin(data_indi,indi_nr_estim)[np.ix_(i_c_z,i_c_lat,i_c_lon)]
    X_train = np.array([X_c_all[0][mask_c_land],
                       X_c_all[1][mask_c_land],
                       X_c_all[2][mask_c_land]]).T

    Y_sigma = field_anom_sig_log[np.ix_(i_c_z,i_c_lat,i_c_lon)][mask_c_land]
    Y_train = np.zeros(Y_sigma.shape)

    ### write static data: x-vals, y-vals, z-vals
    np.save(os.path.join(dir_out,'slope_anom.static'),X_train)

    ### write parameter data: anomaly of alpha (0) and std
    np.save(os.path.join(dir_out,'slope_anom.param.000.000.prior'),np.array([Y_train,Y_sigma]).T)


def setup_sandfrac_anom(settings_gen,settings_run):
    file_surface = os.path.join(settings_gen['dir_clm_surf'],settings_gen['file_clm_surf'])
    sample_every_xy = settings_gen['texture_sample_xy']
    sample_every_z = settings_gen['texture_sample_z']
    dir_out = settings_run['dir_DA']    
    ix_start = settings_gen['texture_start_x']
    iy_start = settings_gen['texture_start_y']
    
    mask_land = xr.open_dataset(file_surface).PFTDATA_MASK.values
    data = xr.open_dataset(file_surface).PCT_SAND

    # select z-indices for which the Kriged fields will be created. For the rest of the indices, interpolate the alpha values
    if sample_every_z is None:
        i_sample_z = np.array([0])
    else:
        i_sample_z = np.arange(data.shape[0]-1,0,-sample_every_z)
        if 0 not in i_sample_z:
            i_sample_z = np.append(i_sample_z,0)
        i_sample_z = np.flip(i_sample_z)
        assert i_sample_z[0] == 0
        assert i_sample_z[-1] == data.shape[0]-1

    ### generate anomaly field, set uniform for now
    # sig_log_standard = 0.046 #set 1 sigma to ~10% deviation
    sig_log_standard = 0.090 # 1 sigma to ~20% deviation
    field_anom = sig_log_standard*np.ones(data.shape)

    ### sample the alpha field at given locations
    i_c_z = i_sample_z
    i_c_lat = np.arange(iy_start,field_anom.shape[1],sample_every_xy)
    i_c_lon = np.arange(ix_start,field_anom.shape[2],sample_every_xy)

    X_c_all = np.meshgrid(i_c_z,i_c_lat,i_c_lon,indexing='ij')
    mask_c_land = mask_land[np.ix_(i_c_lat,i_c_lon)]
    mask_c_land = np.repeat(mask_c_land[np.newaxis,:, :], len(i_sample_z), axis=0).astype(bool)
    X_train = np.array([X_c_all[0][mask_c_land],
                       X_c_all[1][mask_c_land],
                       X_c_all[2][mask_c_land]]).T

    Y_sigma = field_anom[np.ix_(i_c_z,i_c_lat,i_c_lon)][mask_c_land]
    Y_train = np.zeros(Y_sigma.shape)

    ### write static data: x-vals, y-vals, z-vals
    np.save(os.path.join(dir_out,'sandfrac_anom.static'),X_train)

    ### write parameter data: anomaly of alpha (0) and std
    np.save(os.path.join(dir_out,'sandfrac_anom.param.000.000.prior'),np.array([Y_train,Y_sigma]).T)

    
def setup_clayfrac_anom(settings_gen,settings_run):
    file_surface = os.path.join(settings_gen['dir_clm_surf'],settings_gen['file_clm_surf'])
    sample_every_xy = settings_gen['texture_sample_xy']
    sample_every_z = settings_gen['texture_sample_z']
    dir_out = settings_run['dir_DA']    
    ix_start = settings_gen['texture_start_x']
    iy_start = settings_gen['texture_start_y']
    
    mask_land = xr.open_dataset(file_surface).PFTDATA_MASK.values
    data = xr.open_dataset(file_surface).PCT_CLAY

    # select z-indices for which the Kriged fields will be created. For the rest of the indices, interpolate the alpha values
    if sample_every_z is None:
        i_sample_z = np.array([0])
    else:
        i_sample_z = np.arange(data.shape[0]-1,0,-sample_every_z)
        if 0 not in i_sample_z:
            i_sample_z = np.append(i_sample_z,0)
        i_sample_z = np.flip(i_sample_z)
        assert i_sample_z[0] == 0
        assert i_sample_z[-1] == data.shape[0]-1

    ### generate anomaly field, set uniform for now
    # sig_log_standard = 0.046 #set 1 sigma to ~10% deviation
    sig_log_standard = 0.090 # 1 sigma to ~20% deviation
    field_anom = sig_log_standard*np.ones(data.shape)

    ### sample the alpha field at given locations
    i_c_z = i_sample_z
    i_c_lat = np.arange(iy_start,field_anom.shape[1],sample_every_xy)
    i_c_lon = np.arange(ix_start,field_anom.shape[2],sample_every_xy)

    X_c_all = np.meshgrid(i_c_z,i_c_lat,i_c_lon,indexing='ij')
    mask_c_land = mask_land[np.ix_(i_c_lat,i_c_lon)]
    mask_c_land = np.repeat(mask_c_land[np.newaxis,:, :], len(i_sample_z), axis=0).astype(bool)
    X_train = np.array([X_c_all[0][mask_c_land],
                       X_c_all[1][mask_c_land],
                       X_c_all[2][mask_c_land]]).T

    Y_sigma = field_anom[np.ix_(i_c_z,i_c_lat,i_c_lon)][mask_c_land]
    Y_train = np.zeros(Y_sigma.shape)

    ### write static data: x-vals, y-vals, z-vals
    np.save(os.path.join(dir_out,'clayfrac_anom.static'),X_train)

    ### write parameter data: anomaly of alpha (0) and std
    np.save(os.path.join(dir_out,'clayfrac_anom.param.000.000.prior'),np.array([Y_train,Y_sigma]).T)


def setup_orgfrac_anom(settings_gen,settings_run):
    file_surface = os.path.join(settings_gen['dir_clm_surf'],settings_gen['file_clm_surf'])
    sample_every_xy = settings_gen['texture_sample_xy']
    sample_every_z = settings_gen['texture_sample_z']
    dir_out = settings_run['dir_DA']    
    ix_start = settings_gen['texture_start_x']
    iy_start = settings_gen['texture_start_y']
    
    mask_land = xr.open_dataset(file_surface).PFTDATA_MASK.values
    data = xr.open_dataset(file_surface).ORGANIC

    # select z-indices for which the Kriged fields will be created. For the rest of the indices, interpolate the alpha values
    if sample_every_z is None:
        i_sample_z = np.array([0])
    else:
        i_sample_z = np.arange(data.shape[0]-1,0,-sample_every_z)
        if 0 not in i_sample_z:
            i_sample_z = np.append(i_sample_z,0)
        i_sample_z = np.flip(i_sample_z)
        assert i_sample_z[0] == 0
        assert i_sample_z[-1] == data.shape[0]-1

    ### generate anomaly field, set uniform for now
    # sig_log_standard = 0.046 #set 1 sigma to ~10% deviation
    sig_log_standard = 0.090 # 1 sigma to ~20% deviation
    field_anom = sig_log_standard*np.ones(data.shape)

    ### sample the alpha field at given locations
    i_c_z = i_sample_z
    i_c_lat = np.arange(iy_start,field_anom.shape[1],sample_every_xy)
    i_c_lon = np.arange(ix_start,field_anom.shape[2],sample_every_xy)

    X_c_all = np.meshgrid(i_c_z,i_c_lat,i_c_lon,indexing='ij')
    mask_c_land = mask_land[np.ix_(i_c_lat,i_c_lon)]
    mask_c_land = np.repeat(mask_c_land[np.newaxis,:, :], len(i_sample_z), axis=0).astype(bool)
    X_train = np.array([X_c_all[0][mask_c_land],
                       X_c_all[1][mask_c_land],
                       X_c_all[2][mask_c_land]]).T

    Y_sigma = field_anom[np.ix_(i_c_z,i_c_lat,i_c_lon)][mask_c_land]
    Y_train = np.zeros(Y_sigma.shape)

    ### write static data: x-vals, y-vals, z-vals
    np.save(os.path.join(dir_out,'orgfrac_anom.static'),X_train)

    ### write parameter data: anomaly of alpha (0) and std
    np.save(os.path.join(dir_out,'orgfrac_anom.param.000.000.prior'),np.array([Y_train,Y_sigma]).T)

    
def setup_Ks_tensor(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    tensor_val_mean = np.log10(1000)
    tensor_val_std = 1.0 #std = 1 order of magnitude
    np.save(os.path.join(dir_out,'Ks_tensor.param.000.000.prior'),np.array([[tensor_val_mean],[tensor_val_std]]).T)
    

def setup_medlyn_slope(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    vals_mean = np.array([2.3499999 , 2.3499999 , 2.3499999 , 4.11999989,
                       4.11999989, 4.44999981, 4.44999981, 4.44999981, 4.69999981,
                       4.69999981, 4.69999981, 2.22000003, 5.25      , 1.62      ,
                       5.78999996, 5.78999996])

    vals_std = 0.5*np.ones(len(vals_mean)) 
    np.save(os.path.join(dir_out,'medlyn_slope.param.000.000.prior'),np.array([vals_mean,vals_std]).T)
    
    
def setup_medlyn_intercept(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    vals_mean = 100.*np.ones(16)
    vals_std = 50*np.ones(len(vals_mean)) 
    np.save(os.path.join(dir_out,'medlyn_intercept.param.000.000.prior'),np.array([vals_mean,vals_std]).T)
    
   
def setup_fff(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    vals_mean = np.log10(.5)
    vals_std = .5 #2 std: 1 order of magnitude, see Yan et al. (2023)
    np.save(os.path.join(dir_out,'fff.param.000.000.prior'),np.array([[vals_mean],[vals_std]]).T)
     
def setup_b_slope(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    vals_mean = 0.175
    vals_std = 0.012 #see test_hydraulic_relations.ipynb
    np.save(os.path.join(dir_out,'b_slope.param.000.000.prior'),np.array([[vals_mean],[vals_std]]).T)

def setup_b_intercept(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    vals_mean = 2.603
    vals_std = 0.236 #see test_hydraulic_relations.ipynb
    np.save(os.path.join(dir_out,'b_intercept.param.000.000.prior'),np.array([[vals_mean],[vals_std]]).T)
   
def setup_log_psis_slope(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    vals_mean = -0.013
    vals_std = 0.002 #see test_hydraulic_relations.ipynb
    np.save(os.path.join(dir_out,'log_psis_slope.param.000.000.prior'),np.array([[vals_mean],[vals_std]]).T)

def setup_log_psis_intercept(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    vals_mean = 1.873
    vals_std = 0.107 #see test_hydraulic_relations.ipynb
    np.save(os.path.join(dir_out,'log_psis_intercept.param.000.000.prior'),np.array([[vals_mean],[vals_std]]).T)
   
def setup_log_ks_slope(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    vals_mean = 0.015
    vals_std = 0.002 #see test_hydraulic_relations.ipynb
    np.save(os.path.join(dir_out,'log_ks_slope.param.000.000.prior'),np.array([[vals_mean],[vals_std]]).T)

def setup_log_ks_intercept(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    vals_mean = -0.813
    vals_std = 0.102 #see test_hydraulic_relations.ipynb
    np.save(os.path.join(dir_out,'log_ks_intercept.param.000.000.prior'),np.array([[vals_mean],[vals_std]]).T)

def setup_thetas_slope(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    vals_mean = -0.00123
    vals_std = 2.328e-4 #see test_hydraulic_relations.ipynb
    np.save(os.path.join(dir_out,'thetas_slope.param.000.000.prior'),np.array([[vals_mean],[vals_std]]).T)

def setup_thetas_intercept(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    vals_mean = 0.489
    vals_std = 0.012 #see test_hydraulic_relations.ipynb
    np.save(os.path.join(dir_out,'thetas_intercept.param.000.000.prior'),np.array([[vals_mean],[vals_std]]).T)
    
def setup_orgmax(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    vals_mean = 130.
    vals_std = 30. 
    np.save(os.path.join(dir_out,'orgmax.param.000.000.prior'),np.array([[vals_mean],[vals_std]]).T)
  
def setup_om_hydraulic(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    # data['om_thetas_surf'] = 0.93
    # data['om_b_surf'] = 2.7
    # data['om_psis_surf'] = 10.3
    # data['om_ks_surf'] = 0.28
    # data['om_thetas_diff'] = 0.1
    # data['om_b_diff'] = 9.3
    # data['om_psis_diff'] = 0.2
    std_s = np.log10(1.1) #1.100: use 'standard' option of ~10% variation per C.I.
    std_th = np.log10(1.05) #uncertainty for porosity - set such that 95% CI falls below 1.0
    # std_k = 1.042 # this led to way too high k -> crashed simulations
    std_k = 1.042/2
    
    vals_mean = np.log10([0.900, 2.700, 10.30, 0.129, 0.100, 9.300, 0.200])
    vals_std =  np.array([std_th,std_s, std_s, std_k, std_th,std_s, std_s]) 
    
    np.save(os.path.join(dir_out,'om_hydraulic.param.000.000.prior'),np.array([vals_mean,vals_std]).T)
        
    
if __name__ == '__main__':
    
    file_indi = '/p/project/cjibg36/kaandorp2/TSMP_patched/tsmp_cordex_111x108/input_pf/EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_111x108_INDICATOR_regridded_rescaled_SoilGrids250-v2017_BGR3_alv.sa'
    sample_every_xy = 5
    sample_every_z = 3

    setup_Ks(file_indi,sample_every_xy,sample_every_z)