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

if __name__ == '__main__':
    
    file_indi = '/p/project/cjibg36/kaandorp2/TSMP_patched/tsmp_cordex_111x108/input_pf/EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_111x108_INDICATOR_regridded_rescaled_SoilGrids250-v2017_BGR3_alv.sa'
    sample_every_xy = 5
    sample_every_z = 3

    setup_Ks(file_indi,sample_every_xy,sample_every_z)