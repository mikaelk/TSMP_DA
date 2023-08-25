import numpy as np
import sys
extra_path = '/p/project/cjibg36/kaandorp2/Git/SLOTH' 
if extra_path not in sys.path:
    sys.path.append(extra_path)
from sloth.IO import readSa, writeSa
import os


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
    
    
def setup_Ks_tensor(settings_setup,settings_run):
    dir_out = settings_run['dir_DA']
    tensor_val_mean = np.log10(1000)
    tensor_val_std = 1.0 #std = 1 order of magnitude
    np.save(os.path.join(dir_out,'Ks_tensor.param.000.000.prior'),np.array([[tensor_val_mean],[tensor_val_std]]).T)
    
    
if __name__ == '__main__':
    
    file_indi = '/p/project/cjibg36/kaandorp2/TSMP_patched/tsmp_cordex_111x108/input_pf/EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_111x108_INDICATOR_regridded_rescaled_SoilGrids250-v2017_BGR3_alv.sa'
    sample_every_xy = 5
    sample_every_z = 3

    setup_Ks(file_indi,sample_every_xy,sample_every_z)