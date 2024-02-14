import numpy as np
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import os
import xarray as xr
import shutil
from helpers import readSa, writeSa


def generate_anomaly_field(X_train,Y_train,data_indi,mode='ml',
                           shape_out=None,mask_land=None,length_scale=30,length_scale_bounds='fixed',
                           nu=1.5,vary_depth=True,depth_setting='PFL'):
    """
    Generate anomaly field using Gaussian Process Regression
    
    X_train: [X,Y,Z] coordinates
    Y_train: corresponding response
    data_indi: indicator field, used to mask ocean
    mode: most likely (ml) or additional random noise based on Kriged function. I only used ml in the end
    shape_out: prescribe output shape
    mask_land: can be used to specify land mask (instead of getting this from the indicator file)
    length_scale: length scale of the RBF kernel
    nu: parameter of the RBF kernel (1.5=exponential kernel)
    vary_depth: I used False in the end for the CLM simulations, which makes the perturbations 2D (i.e. depth invariant)
    depth_setting: PFL or eCLM; used to specify the layer depths. Not required when setting vary_depth=False
    """
    if shape_out:
        anom_field = np.nan*np.zeros(shape_out)
    else: #base anomaly field on the indicator file shape
        anom_field = np.nan*np.zeros(data_indi.shape)
    
    def fit_gprs(i_sample_z,X_train,Y_train):

        gprs_ = {} 

        for i1, i_sample in enumerate(i_sample_z): #only train the GPR on several layers
            mask = X_train[:,0] == i_sample
            X_train_ = X_train[mask,1:]
            Y_train_ = Y_train[mask]
            # Y_sigma_ = Y_train[mask,1]
            # gprs_[i_sample] = GaussianProcessRegressor(kernel=kernel,alpha=1.*(Y_sigma_**2))
            gprs_[i_sample] = GaussianProcessRegressor(kernel=kernel)
            gprs_[i_sample].fit(X_train_, Y_train_)

        return gprs_

    kernel = 1.0 * Matern(length_scale=length_scale,nu=nu,length_scale_bounds=length_scale_bounds)

    if depth_setting == 'PFL':
        indi_dz = 2*np.array([9.0,7.5,5.0,5.0,2.0,0.5,0.35,0.25,0.15,0.10,0.065,0.035,0.025,0.015,0.01])
        indi_depths = np.flip(np.cumsum(np.flip(indi_dz)))
    elif depth_setting == 'eCLM':
        eCLM_dz = np.array([0.02,0.04,0.06,0.08,0.120,
                    0.160,0.200,0.240,0.280,0.320,
                    0.360,0.400,0.440,0.540,0.640,
                    0.740,0.840,0.940,1.040,1.140,
                    2.390,4.676,7.635,11.140,15.115])
        indi_depths = np.cumsum(eCLM_dz) - .5*eCLM_dz
    else:
        raise RuntimeError('define dz')

    # indices of layers to be sampled by Kriging
    i_sample_z = np.unique(X_train[:,0])
    # fit the Gaussian Process Regressors
    gprs = fit_gprs(i_sample_z,X_train,Y_train)

    X = np.meshgrid(np.arange(0,anom_field.shape[1]),
                    np.arange(0,anom_field.shape[2]),indexing='ij')

    i_lower_old = np.inf

    for i_layer in range(anom_field.shape[0]):
        
        if mask_land is None:     
            mask_land = (data_indi[i_layer,:] < 20)

        X_test = np.array([X[0][mask_land],
                        X[1][mask_land]]).T

        #2D mode -> same anomaly for all layers
        if not vary_depth: 
            if i_layer == 0:
                anom_ = gprs[i_layer].predict(X_test)
                for i_layer_ in range(anom_field.shape[0]):
                    anom_field[i_layer_,mask_land] = anom_
            else: #other layers are based on layer 0 -> pass
                pass
            
        # 3D mode: layer is sampled -> use Kriging to interpolate horizontally
        elif i_layer in i_sample_z:
            if mode == 'ml':
                anom_ = gprs[i_layer].predict(X_test)
            elif mode == 'rnd':
                anom_ = gprs[i_layer].sample_y(X_test,random_state=rnd_seed)[:,0]
            else:
                raise RuntimeError('Mode should be ml (most likely) or rnd (random)')
            anom_field[i_layer,mask_land] = anom_

        # 3D mode: layer is not sampled -> interpolate in depth linearly
        else:
            i_lower = i_sample_z[np.digitize(i_layer,i_sample_z) - 1]
            i_upper = i_sample_z[np.digitize(i_layer,i_sample_z)]

            if i_lower == i_lower_old:
                # predictions for upper/lower fields already made, skip
                pass
            else:
                if mode == 'ml':
                    anom_upper = gprs[i_upper].predict(X_test)
                    anom_lower = gprs[i_lower].predict(X_test)
                elif mode == 'rnd':
                    anom_upper = gprs[i_upper].sample_y(X_test,random_state=rnd_seed)[:,0]
                    anom_lower = gprs[i_lower].sample_y(X_test,random_state=rnd_seed)[:,0]

                i_lower_old = i_lower

            z_upper = indi_depths[i_upper]
            z_lower = indi_depths[i_lower]
            z_interp = indi_depths[i_layer]

            assert(z_upper < z_lower)
            assert(z_interp > z_upper)

            anom_interp = ( ((z_interp - z_upper) / (z_lower - z_upper)) * anom_lower) + ( ((z_lower - z_interp) / (z_lower - z_upper)) * anom_upper)
            anom_field[i_layer,mask_land] = anom_interp    

    return anom_field

#############################################################################################
#### Functions to generate perturbation fields for soil texture: sand, clay, organic matter #
#############################################################################################

def generate_sandfrac_anom(i_real,settings_gen,settings_run):
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    dir_in = settings_run['dir_DA']   

    dir_surface = settings_gen['dir_clm_surf']
    filename_surface = settings_gen['file_clm_surf']
    file_out = os.path.join(dir_real,os.path.basename(filename_surface)) 
    if os.path.exists(file_out): #if the surface file is already existing in DA dir, open it
        dir_surface = dir_real
    
    sample_every_xy = settings_gen['texture_sample_xy']
    sample_every_z = settings_gen['texture_sample_z']
    if sample_every_z is None:
        vary_depth=False
    else:
        vary_depth=True
        
    mode = 'rnd'
    length_scale = settings_gen['texture_anom_l']
    length_scale_bounds = settings_gen['texture_anom_l_bounds']
    nu = settings_gen['texture_nu']
    
    plot = settings_gen['texture_plot']
    folder_figs = os.path.join(dir_real,'figures') 

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
    if mode == 'rnd':
        rnd_seed = np.random.randint(100)

    ### 1) read texture data
    data = xr.open_dataset(os.path.join(dir_surface,filename_surface))
    mask_land = (data.PFTDATA_MASK.values == 1).astype(bool)
    data_texture = data.PCT_SAND.values
    data_clay = data.PCT_CLAY.values
    
    ### 2) Add anomaly field
    X_train = np.load(os.path.join(dir_in,'sandfrac_anom.static.npy'))
    Y_train = np.load(os.path.join(dir_in,'sandfrac_anom.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))

    anom_field = generate_anomaly_field(X_train,Y_train,data_texture,mode='ml',
                                        shape_out=data_texture.shape,mask_land=mask_land,
                                        length_scale=length_scale,
                                        length_scale_bounds=length_scale_bounds,
                                        nu=nu,
                                        vary_depth=vary_depth,depth_setting='eCLM')
    anom_field[:,~mask_land]=0

    data_texture = data_texture*(10**(anom_field)) # anomaly field: 10**-1 -> 10**0 -> 10**1: 0.1 -> 1 -> 10 (multiplicative factor)

    # make sure sand+clay do not exceed 100%
    total = data_texture+data_clay
    data_texture[total>100] -= (total-100)[total>100]
    
    # max_sand = 99. 
    # data_texture[data_texture>max_sand] = max_sand

    data['PCT_SAND'] = (['nlevsoi','lsmlat','lsmlon'],data_texture)

    data.to_netcdf(file_out)

    if plot:
        plot_ctrl = True
        if not os.path.exists(folder_figs):
            print('Making folder for verification plots in %s' % folder_figs )
            os.mkdir(folder_figs)

        plt.figure()
        plt.pcolormesh(data_texture[0,:,:] )
        if plot_ctrl:
            plt.plot(X_train[:,2],X_train[:,1],'kx')
        plt.colorbar(extend='both')
        plt.title('Sand fraction (upper layer)')
        plt.savefig( os.path.join(folder_figs,'sandfrac_real%3.3i.png'% (i_real)) )
        plt.close()  

        plt.figure()
        plt.pcolormesh(data_texture[-1,:,:] )
        if plot_ctrl:
            plt.plot(X_train[:,2],X_train[:,1],'kx')
        plt.colorbar(extend='both')
        plt.title('Sand fraction (lowest layer)')
        plt.savefig( os.path.join(folder_figs,'sandfrac_real%3.3i_deep.png'% (i_real)) )
        plt.close()  

def generate_clayfrac_anom(i_real,settings_gen,settings_run):
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    dir_in = settings_run['dir_DA']   

    dir_surface = settings_gen['dir_clm_surf']
    filename_surface = settings_gen['file_clm_surf']
    file_out = os.path.join(dir_real,os.path.basename(filename_surface)) 
    file_out_tmp = file_out + '.tmp'
    if os.path.exists(file_out): #if the surface file is already existing in DA dir, open it
        dir_surface = dir_real
    
    sample_every_xy = settings_gen['texture_sample_xy']
    sample_every_z = settings_gen['texture_sample_z']
    if sample_every_z is None:
        vary_depth=False
    else:
        vary_depth=True
        
    mode = 'rnd'
    length_scale = settings_gen['texture_anom_l']
    length_scale_bounds = settings_gen['texture_anom_l_bounds']
    nu = settings_gen['texture_nu']

    plot = settings_gen['texture_plot']
    folder_figs = os.path.join(dir_real,'figures') 

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
    if mode == 'rnd':
        rnd_seed = np.random.randint(100)

    ### 1) read texture data
    data = xr.open_dataset(os.path.join(dir_surface,filename_surface))
    mask_land = (data.PFTDATA_MASK.values == 1).astype(bool)
    data_texture = data.PCT_CLAY.values
    data_sand = data.PCT_SAND.values
    
    ### 2) Add anomaly field
    X_train = np.load(os.path.join(dir_in,'clayfrac_anom.static.npy'))
    Y_train = np.load(os.path.join(dir_in,'clayfrac_anom.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))

    anom_field = generate_anomaly_field(X_train,Y_train,data_texture,mode='ml',
                                        shape_out=data_texture.shape,mask_land=mask_land,
                                        length_scale=length_scale,
                                        length_scale_bounds=length_scale_bounds,
                                        nu=nu,
                                        vary_depth=vary_depth,depth_setting='eCLM')
    anom_field[:,~mask_land]=0

    data_texture = data_texture*(10**(anom_field)) # anomaly field: 10**-1 -> 10**0 -> 10**1: 0.1 -> 1 -> 10 (multiplicative factor)

    # make sure sand+clay do not exceed 100%
    total = data_texture+data_sand
    data_texture[total>100] -= (total-100)[total>100]

    if np.any(data_texture<0):
        print('Warning: Clay percentage below 0!!!',flush=True)
        data_texture[data_texture<0] = 0
        
    # max_sand = 99. 
    # data_texture[data_texture>max_sand] = max_sand

    data['PCT_CLAY'] = (['nlevsoi','lsmlat','lsmlon'],data_texture)

    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)
    # os.remove(file_out_tmp)
    
    if plot:
        plot_ctrl = True
        if not os.path.exists(folder_figs):
            print('Making folder for verification plots in %s' % folder_figs )
            os.mkdir(folder_figs)

        plt.figure()
        plt.pcolormesh(data_texture[0,:,:] )
        if plot_ctrl:
            plt.plot(X_train[:,2],X_train[:,1],'kx')
        plt.colorbar(extend='both')
        plt.title('Clay fraction (upper layer)')
        plt.savefig( os.path.join(folder_figs,'clayfrac_real%3.3i.png'% (i_real)) )
        plt.close()  
        
    
def generate_orgfrac_anom(i_real,settings_gen,settings_run):
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    dir_in = settings_run['dir_DA']   

    dir_surface = settings_gen['dir_clm_surf']
    filename_surface = settings_gen['file_clm_surf']
    file_out = os.path.join(dir_real,os.path.basename(filename_surface)) 
    file_out_tmp = file_out + '.tmp'
    if os.path.exists(file_out): #if the surface file is already existing in DA dir, open it
        dir_surface = dir_real
    
    sample_every_xy = settings_gen['texture_sample_xy']
    sample_every_z = settings_gen['texture_sample_z']
    if sample_every_z is None:
        vary_depth=False
    else:
        vary_depth=True
        
    mode = 'rnd'
    length_scale = settings_gen['texture_anom_l']
    length_scale_bounds = settings_gen['texture_anom_l_bounds']
    nu = settings_gen['texture_nu']
    
    plot = settings_gen['texture_plot']
    folder_figs = os.path.join(dir_real,'figures') 

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
    if mode == 'rnd':
        rnd_seed = np.random.randint(100)

    ### 1) read texture data
    data = xr.open_dataset(os.path.join(dir_surface,filename_surface))
    mask_land = (data.PFTDATA_MASK.values == 1).astype(bool)
    data_texture = data.ORGANIC.values
    data_sand =  data.PCT_SAND.values
    data_clay =  data.PCT_CLAY.values
    
    ### 2) Add anomaly field
    X_train = np.load(os.path.join(dir_in,'orgfrac_anom.static.npy'))
    Y_train = np.load(os.path.join(dir_in,'orgfrac_anom.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))

    anom_field = generate_anomaly_field(X_train,Y_train,data_texture,mode='ml',
                                        shape_out=data_texture.shape,mask_land=mask_land,
                                        length_scale=length_scale,
                                        length_scale_bounds=length_scale_bounds,
                                        nu=nu,
                                        vary_depth=vary_depth,depth_setting='eCLM')
    anom_field[:,~mask_land]=0

    data_texture = data_texture*(10**(anom_field)) # anomaly field: 10**-1 -> 10**0 -> 10**1: 0.1 -> 1 -> 10 (multiplicative factor)
    
    # check: organic matter density should not exceed 130 (see CLM5 parameter file: organic_max)
    # perhaps this should be changed to read in the orgmax parameter instead, but then a check needs to be included if this parameter is actually assimilated or not
    max_organic = 130.
    data_texture[data_texture>max_organic] = max_organic
    
    data['ORGANIC'] = (['nlevsoi','lsmlat','lsmlon'],data_texture)

    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)
    # os.remove(file_out_tmp)

    if plot:
        plot_ctrl = True
        if not os.path.exists(folder_figs):
            print('Making folder for verification plots in %s' % folder_figs )
            os.mkdir(folder_figs)

        plt.figure()
        plt.pcolormesh(data_texture[0,:,:] )
        if plot_ctrl:
            plt.plot(X_train[:,2],X_train[:,1],'kx')
        plt.colorbar(extend='both')
        plt.title('Organic density (upper layer)')
        plt.savefig( os.path.join(folder_figs,'orgfrac_real%3.3i.png'% (i_real)) )
        plt.close()  
        
############################################################################################
#### Functions to generate CLM parameters
############################################################################################

def generate_medlyn_slope(i_real,settings_gen,settings_run):
    '''
    Old medlyn functions; use _v2 version instead
    '''
    dir_in = settings_run['dir_DA']
    values = np.load(os.path.join(dir_in,'medlyn_slope.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))
    values[values<.3] = .3 #slope needs to be positive, set to low value when necessary
    
    dir_setup = settings_run['dir_setup']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    file_in = os.path.join(dir_setup,'input_clm/clm5_params.c171117.nc')
    file_out = os.path.join(dir_real,'clm5_params.c171117.nc') 
    file_out_tmp = file_out + '.tmp' 
    
    if os.path.exists(file_out): #the param file has been adjusted already -> open it again
        file_in = file_out
    
    data = xr.load_dataset(file_in)
    data.medlynslope.values[1:len(values)+1] = values.ravel()
    
    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)    
    
def generate_medlyn_intercept(i_real,settings_gen,settings_run):
    '''
    Old medlyn functions; use _v2 version instead
    '''    
    dir_in = settings_run['dir_DA']
    values_log10 = np.load(os.path.join(dir_in,'medlyn_intercept.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))
    values = 10**values_log10
    values[values<.1] = .1 #negative intercept -> negative stomatal conductance, set to low value instead
    
    dir_setup = settings_run['dir_setup']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    file_in = os.path.join(dir_setup,'input_clm/clm5_params.c171117.nc')
    file_out = os.path.join(dir_real,'clm5_params.c171117.nc') 
    file_out_tmp = file_out + '.tmp' 
    
    if os.path.exists(file_out): #the param file has been adjusted already -> open it again
        file_in = file_out
    
    data = xr.load_dataset(file_in)
    data.medlynintercept[1:len(values)+1] = values.ravel()
    
    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)   
    
    
def generate_medlyn_slope_v2(i_real,settings_gen,settings_run):
    
    dir_in = settings_run['dir_DA']
    values = np.load(os.path.join(dir_in,'medlyn_slope_v2.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))
    values[values<.3] = .3 #slope needs to be positive, set to low value when necessary
    
    dir_setup = settings_run['dir_setup']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    file_in = os.path.join(dir_setup,'input_clm/clm5_params.c171117.nc')
    file_out = os.path.join(dir_real,'clm5_params.c171117.nc') 
    file_out_tmp = file_out + '.tmp' 
    
    if os.path.exists(file_out): #the param file has been adjusted already -> open it again
        file_in = file_out
    
    data = xr.load_dataset(file_in)
    i_pfts = np.array([1,3,5,7,9,10,13,15,16]) #active pfts in the clm setup
    data.medlynslope[i_pfts] = values.ravel()
    
    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)    
    
def generate_medlyn_intercept_v2(i_real,settings_gen,settings_run):
    
    dir_in = settings_run['dir_DA']
    values_log10 = np.load(os.path.join(dir_in,'medlyn_intercept_v2.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))
    values = 10**values_log10
    values[values<.1] = .1 #negative intercept -> negative stomatal conductance, set to low value instead
    
    dir_setup = settings_run['dir_setup']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    file_in = os.path.join(dir_setup,'input_clm/clm5_params.c171117.nc')
    file_out = os.path.join(dir_real,'clm5_params.c171117.nc') 
    file_out_tmp = file_out + '.tmp' 
    
    if os.path.exists(file_out): #the param file has been adjusted already -> open it again
        file_in = file_out
    
    data = xr.load_dataset(file_in)
    i_pfts = np.array([1,3,5,7,9,10,13,15,16]) #active pfts in the clm setup
    data.medlynintercept[i_pfts] = values.ravel()
    
    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)   
    
    
def generate_fff(i_real,settings_gen,settings_run):
    '''
    the fff, or f_sat,decay parameter
    '''
    dir_in = settings_run['dir_DA']
    value_log10 = np.load(os.path.join(dir_in,'fff.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))[0]
    value = 10**value_log10
    
    dir_setup = settings_run['dir_setup']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    file_in = os.path.join(dir_setup,'input_clm/clm5_params.c171117.nc')
    file_out = os.path.join(dir_real,'clm5_params.c171117.nc') 
    file_out_tmp = file_out + '.tmp' 
    
    if os.path.exists(file_out): #the param file has been adjusted already -> open it again
        file_in = file_out
    
    data = xr.load_dataset(file_in)
    data['fff'] = value
    
    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)   
    
    
def generate_mineral_hydraulic(i_real,settings_gen,settings_run):
    '''
    Mineral hydraulic parameters, based on Cosby 1984
    # b slope/intercept, [0 , 1]
    # psis s/i,          [2,  3]
    # ks s/i,            [4,  5]
    # thetas s/i         [6,  7]   
    '''
    dir_in = settings_run['dir_DA']
    values = np.load(os.path.join(dir_in,'mineral_hydraulic.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))
   
    if values[0] < 0.01:
        print('Warning: b_slope moved towards negative values')
        values[0] = 0.01
    if values[7] < 0.01:
        print('Warning: thetas intercept moved towards negative values')
        values[7] = 0.01
    if values[7] > .99:
        print('Warning: thetas intercept moved towards values >1')
        values[7] = .99
    
    dir_setup = settings_run['dir_setup']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    file_in = os.path.join(dir_setup,'input_clm/clm5_params.c171117.nc')
    file_out = os.path.join(dir_real,'clm5_params.c171117.nc') 
    file_out_tmp = file_out + '.tmp' 
    
    if os.path.exists(file_out): #the param file has been adjusted already -> open it again
        file_in = file_out
    
    data = xr.load_dataset(file_in)
    data['b_slope'] = values[0]
    data['b_intercept'] = values[1]
    data['log_psis_slope'] = values[2]
    data['log_psis_intercept'] = values[3]
    data['log_ks_slope'] = values[4]
    data['log_ks_intercept'] = values[5]
    data['thetas_slope'] = values[6]
    data['thetas_intercept'] = values[7]
    
    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)   
    
def generate_orgmax(i_real,settings_gen,settings_run):
    '''
    The maximum organic density; rho_sc,max 
    Old version, use _v2 instead
    '''    
    dir_in = settings_run['dir_DA']
    value = np.load(os.path.join(dir_in,'orgmax.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))[0]
    if value < 10:
        print('Warning: orgmax very small')
        value = 10.
    if value > 300:
        print('Warning: orgmax very large')
        value = 300.
        
    dir_setup = settings_run['dir_setup']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    file_in = os.path.join(dir_setup,'input_clm/clm5_params.c171117.nc')
    file_out = os.path.join(dir_real,'clm5_params.c171117.nc') 
    file_out_tmp = file_out + '.tmp' 
    
    if os.path.exists(file_out): #the param file has been adjusted already -> open it again
        file_in = file_out
    
    data = xr.load_dataset(file_in)
    data['organic_max'].values = np.array([value])
    
    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)  
   
def generate_orgmax_v2(i_real,settings_gen,settings_run):
    '''
    The maximum organic density; rho_sc,max 
    Perturbation %-based

    '''    
    dir_in = settings_run['dir_DA']
    value_log10 = np.load(os.path.join(dir_in,'orgmax_v2.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))[0]
    value = 10**value_log10
    
    if value < 10:
        print('Warning: orgmax very small')
        value = 10.
    if value > 300:
        print('Warning: orgmax very large')
        value = 300.
        
    dir_setup = settings_run['dir_setup']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    file_in = os.path.join(dir_setup,'input_clm/clm5_params.c171117.nc')
    file_out = os.path.join(dir_real,'clm5_params.c171117.nc') 
    file_out_tmp = file_out + '.tmp' 
    
    if os.path.exists(file_out): #the param file has been adjusted already -> open it again
        file_in = file_out
    
    data = xr.load_dataset(file_in)
    data['organic_max'].values = np.array([value])
    
    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)  
    
def generate_om_hydraulic(i_real,settings_gen,settings_run):
    '''
    Organic hydraulic parameters, based on Letts et al. 2000
    # data['om_thetas_surf'] = 0.93
    # data['om_b_surf'] = 2.7
    # data['om_psis_surf'] = 10.3
    # data['om_ks_surf'] = 0.28
    # data['om_thetas_diff'] = 0.1
    # data['om_b_diff'] = 9.3
    # data['om_psis_diff'] = 0.2
    '''
    
    dir_in = settings_run['dir_DA']
    values_ = np.load(os.path.join(dir_in,'om_hydraulic.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))
    values = 10**values_
    
    values[0] = min(values[0],.99) #porosity: upper bound
    values[4] = min(values[4],(values[0]-0.01)) #porosity: lower bound
    values[1] = max(values[1],1.1) # lower bound b
    values[3] = min(values[3],3.) #upper bound for k, high values lead to simulations stopping
    
    dir_setup = settings_run['dir_setup']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    file_in = os.path.join(dir_setup,'input_clm/clm5_params.c171117.nc')
    file_out = os.path.join(dir_real,'clm5_params.c171117.nc') 
    file_out_tmp = file_out + '.tmp' 
    
    if os.path.exists(file_out): #the param file has been adjusted already -> open it again
        file_in = file_out
    
    data = xr.load_dataset(file_in)

    data['om_thetas_surf'] = values[0]
    data['om_b_surf'] = values[1]
    data['om_psis_surf'] = values[2]
    data['om_ks_surf'] = values[3]
    data['om_thetas_diff'] = values[4]
    data['om_b_diff'] = values[5]
    data['om_psis_diff'] = values[6]
    
    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)    
    
    
def generate_h2o_canopy_max(i_real,settings_gen,settings_run):
    '''
    Max canopy water storage
    '''
    dir_in = settings_run['dir_DA']
    value_log10 = np.load(os.path.join(dir_in,'h2o_canopy_max.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))[0]
    value = 10**value_log10
    
    dir_setup = settings_run['dir_setup']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    file_in = os.path.join(dir_setup,'input_clm/clm5_params.c171117.nc')
    file_out = os.path.join(dir_real,'clm5_params.c171117.nc') 
    file_out_tmp = file_out + '.tmp' 
    
    if os.path.exists(file_out): #the param file has been adjusted already -> open it again
        file_in = file_out
    
    data = xr.load_dataset(file_in)
    data['h2o_canopy_max'] = value
    
    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)  
    
def generate_kmax(i_real,settings_gen,settings_run):
    '''
    old kmax, use _v2 instead
    '''    
    dir_in = settings_run['dir_DA']
    values_log10 = np.load(os.path.join(dir_in,'kmax.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))
    values = 10**values_log10
    
    dir_setup = settings_run['dir_setup']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    file_in = os.path.join(dir_setup,'input_clm/clm5_params.c171117.nc')
    file_out = os.path.join(dir_real,'clm5_params.c171117.nc') 
    file_out_tmp = file_out + '.tmp' 
    
    if os.path.exists(file_out): #the param file has been adjusted already -> open it again
        file_in = file_out
    
    data = xr.load_dataset(file_in)
    
    for i_pft in np.arange(1,len(values)+1):
        data.kmax[:,i_pft] = values[i_pft-1]
    
    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)   
    
def generate_kmax_v2(i_real,settings_gen,settings_run):
    '''
    kmax
    '''    
    dir_in = settings_run['dir_DA']
    values_log10 = np.load(os.path.join(dir_in,'kmax_v2.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))
    values = 10**values_log10
    
    dir_setup = settings_run['dir_setup']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    file_in = os.path.join(dir_setup,'input_clm/clm5_params.c171117.nc')
    file_out = os.path.join(dir_real,'clm5_params.c171117.nc') 
    file_out_tmp = file_out + '.tmp' 
    
    if os.path.exists(file_out): #the param file has been adjusted already -> open it again
        file_in = file_out
    
#     for i_pft in np.arange(1,len(values)+1):
#         data.kmax[:,i_pft] = values[i_pft-1]
    
    data = xr.load_dataset(file_in)
    i_pfts = np.array([1,3,5,7,9,10,13,15,16]) #active pfts in the clm setup
    for i_,i_pft in enumerate(i_pfts):
        data.kmax[:,i_pft] = values[i_]
    
    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)   

    
def generate_luna(i_real,settings_gen,settings_run):
    '''
    photosynthesis parameters for the luna module in CLM5
    Perturb Jmaxb0, Jmaxb1, t_cj0 (Wc2Wjb0 in CLM), H (relhExp in CLM)
    '''  
    dir_in = settings_run['dir_DA']
    values_log10 = np.load(os.path.join(dir_in,'luna.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))
    values = 10**values_log10
    
    dir_setup = settings_run['dir_setup']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    file_in = os.path.join(dir_setup,'input_clm/clm5_params.c171117.nc')
    file_out = os.path.join(dir_real,'clm5_params.c171117.nc') 
    file_out_tmp = file_out + '.tmp' 
    
    if os.path.exists(file_out): #the param file has been adjusted already -> open it again
        file_in = file_out
    
    data = xr.load_dataset(file_in)
    
    data['Jmaxb0'] = values[0]
    data['Jmaxb1'] = values[1]
    data['Wc2Wjb0'] = values[2]
    data['relhExp'] = values[3]
    
    data.to_netcdf(file_out_tmp)
    data.close()
    shutil.move(file_out_tmp,file_out)   
    
    
################################################################################################################################################################    
####### Some old ParFlow functions that are not used for the CLM DA. These are old so probably don't work, but I'll leave them here in case they are useful in the future
################################################################################################################################################################    

def generate_Ks_tensor(i_real,settings_gen,settings_run):
    
    dir_in = settings_run['dir_DA']
    tensor_value_log10 = np.load(os.path.join(dir_in,'Ks_tensor.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))
    tensor_value = 10**tensor_value_log10
    
    dir_setup = settings_run['dir_setup']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    file_out = os.path.join(dir_real,'coup_oas.tcl') 
    
    # replace value in file
    with open( os.path.join(dir_setup,'namelists/coup_oas.tcl') , 'r') as file :
        filedata = file.read()

    filedata = filedata.replace('__Ks_tensor__', '%.1f'%tensor_value)

    with open(file_out, 'w') as file:
        file.write(filedata)
    os.chmod(file_out,0o755)    
    
def generate_Ks(i_real,settings_gen,settings_run):

    file_indi = settings_gen['file_indi']
    # file_Ks_out = settings_gen['Ks_file_out']
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    file_Ks_out = os.path.join(dir_real,'Ks.sa') 
    mode = settings_gen['Ks_mode']
    plot = settings_gen['Ks_plot']
    folder_figs = os.path.join(dir_real,'figures') 
    dir_in = settings_run['dir_DA']
    
    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
    if mode == 'rnd':
        rnd_seed = np.random.randint(100)
   
    def fit_gprs(i_sample_z,X_train,Y_train):
    
        gprs_ = {} 

        for i1, i_sample in enumerate(i_sample_z): #only train the GPR on several layers
            mask = X_train[:,0] == i_sample
            X_train_ = X_train[mask,1:]
            Y_train_ = Y_train[mask]
            # Y_sigma_ = Y_train[mask,1]
            # gprs_[i_sample] = GaussianProcessRegressor(kernel=kernel,alpha=1.*(Y_sigma_**2))
            gprs_[i_sample] = GaussianProcessRegressor(kernel=kernel)
            gprs_[i_sample].fit(X_train_, Y_train_)

        return gprs_
    
    ### Standard parameters that can be changed
    Ks_water = -5 #log10 value 
    Ks_allv = False #log10 value (-1) or False
    indi_water = 21
    indi_allv = 13

    X_train = np.load(os.path.join(dir_in,'Ks.static.npy'))
    Y_train = np.load(os.path.join(dir_in,'Ks.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))
    # Y_sigma = np.load(os.path.join(dir_in,'Ks.param.%3.3i.%3.3i.000.npy'%(settings_gen['i_date'],settings_gen['i_iter']) ))[:,1]
        
    kernel = 1.0 * Matern(length_scale=30,nu=1.5,length_scale_bounds='fixed')
    # kernel = 1.0 * Matern(length_scale=20,nu=1.5)

    data_indi = readSa(file_indi)
    indi_dz = 2*np.array([9.0,7.5,5.0,5.0,2.0,0.5,0.35,0.25,0.15,0.10,0.065,0.035,0.025,0.015,0.01])
    indi_depths = np.flip(np.cumsum(np.flip(indi_dz)))
    ### /parameters

    # indices of layers to be sampled by Kriging
    i_sample_z = np.unique(X_train[:,0])
    # fit the Gaussian Process Regressors
    gprs = fit_gprs(i_sample_z,X_train,Y_train)

    Ks_field = np.nan*np.zeros(data_indi.shape)

    X = np.meshgrid(np.arange(0,Ks_field.shape[1]),
                    np.arange(0,Ks_field.shape[2]),indexing='ij')

    i_lower_old = np.inf
    Ks_field_upper = np.nan*np.zeros(X[0].shape)
    Ks_field_lower = np.nan*np.zeros(X[0].shape)

    for i_layer in range(Ks_field.shape[0]):
        print('Generating Ks for layer %i' % i_layer,flush=True)
        mask_land = (data_indi[i_layer,:] < 20)
        X_test = np.array([X[0][mask_land],
                        X[1][mask_land]]).T


        # layer is sampled -> Kriging
        if i_layer in i_sample_z:
            if mode == 'ml':
                Ks_ = gprs[i_layer].predict(X_test)
            elif mode == 'rnd':
                Ks_ = gprs[i_layer].sample_y(X_test,random_state=rnd_seed)[:,0]
            else:
                raise RuntimeError('Mode should be ml (most likely) or rnd (random)')
            Ks_field[i_layer,mask_land] = Ks_

        # layer is not sampled -> interpolate linearly
        else:
            i_lower = i_sample_z[np.digitize(i_layer,i_sample_z) - 1]
            i_upper = i_sample_z[np.digitize(i_layer,i_sample_z)]

            if i_lower == i_lower_old:
                # predictions for upper/lower fields already made, skip
                pass
            else:
                if mode == 'ml':
                    Ks_upper = gprs[i_upper].predict(X_test)
                    Ks_lower = gprs[i_lower].predict(X_test)
                elif mode == 'rnd':
                    Ks_upper = gprs[i_upper].sample_y(X_test,random_state=rnd_seed)[:,0]
                    Ks_lower = gprs[i_lower].sample_y(X_test,random_state=rnd_seed)[:,0]

                Ks_field_upper[mask_land] = Ks_upper
                Ks_field_lower[mask_land] = Ks_lower
                i_lower_old = i_lower

            z_upper = indi_depths[i_upper]
            z_lower = indi_depths[i_lower]
            z_interp = indi_depths[i_layer]

            assert(z_upper < z_lower)
            assert(z_interp > z_upper)

            Ks_interp = ( ((z_interp - z_upper) / (z_lower - z_upper)) * Ks_lower) + ( ((z_lower - z_interp) / (z_lower - z_upper)) * Ks_upper)
            Ks_field[i_layer,mask_land] = Ks_interp    

    Ks_field[data_indi >= indi_water] = Ks_water
    if Ks_allv:
        Ks_field[data_indi == indi_allv] = Ks_allv
    Ks_field = 10**Ks_field

    writeSa(file_Ks_out,Ks_field)

    if plot:
        if not os.path.exists(folder_figs):
            print('Making folder for verification plots in %s' % folder_figs )
            os.mkdir(folder_figs)

        for i_layer in range(Ks_field.shape[0]):
            plt.figure()
            plt.pcolormesh(Ks_field[i_layer,:,:],norm=colors.LogNorm(vmin=1e-5, vmax=.5) )
            plt.colorbar()
            if i_layer in i_sample_z:
                method_ = 'Kriged'
            else:
                method_ = 'Interpolated'
            plt.title('Permeability, %s, layer:%i [m/h]' % (method_,i_layer) )
            plt.savefig( os.path.join(folder_figs,'Ks_real%3.3i_%3.3i.png'% (i_real,i_layer)) )
            plt.close()

            
def generate_Ks_anom(i_real,settings_setup,settings_run):
  
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    file_Ks_out = os.path.join(dir_real,'Ks.sa') 
    
    file_indi = settings_setup['file_indi']
    mode = settings_setup['Ks_mode']
    plot = settings_setup['Ks_plot']
    folder_figs = os.path.join(dir_real,'figures') 
    dir_in = settings_run['dir_DA']
    
    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
    if mode == 'rnd':
        rnd_seed = np.random.randint(100)
   

    ### 1) indicator file data
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

    ### generate Ks field from indicator data
    field_Ks_log = np.zeros(data_indi.shape)

    for i1,indicator in enumerate(indi_nr):
        mask = (data_indi == indicator)

        if indicator <= 12: #surface layers, soilgrids
            # use Rosetta: values are given in log10(cm/day) -> convert to log10(m/h) for ParFlow
            field_Ks_log[mask] = indi_logKs[i1] - np.log10(100*24) #go from cm/day (Rosetta) to m/h (ParFlow)
        elif indicator < 20: # deep subsurface
            # use standard ParFlow values. Take the log10, and use predefined uncertainty
            field_Ks_log[mask] = np.log10(indi_perm[i1])
        else: #lake and ocean
            field_Ks_log[mask] = np.log10(indi_perm[i1])
    assert (field_Ks_log==0).sum() == 0
    
    
    ### 2) Add anomaly field
    anom_water = 0 # no anomaly for water/alluvium
    anom_allv = 0 #
    indi_water = 21
    indi_allv = 13
    
    X_train = np.load(os.path.join(dir_in,'Ks_anom.static.npy'))
    Y_train = np.load(os.path.join(dir_in,'Ks_anom.param.%3.3i.%3.3i.%3.3i.npy'%(settings_setup['i_date'],settings_setup['i_iter'],i_real) ))
      
    anom_field = generate_anomaly_field(X_train,Y_train,data_indi,mode='ml')
    
    anom_field[data_indi >= indi_water] = anom_water
    anom_field[data_indi == indi_allv] = anom_allv
   
    Ks_field = 10**(field_Ks_log + anom_field)
 
    writeSa(file_Ks_out,Ks_field)

    if plot:
        i_sample_z = np.unique(X_train[:,0])
        
        if not os.path.exists(folder_figs):
            print('Making folder for verification plots in %s' % folder_figs )
            os.mkdir(folder_figs)

        for i_layer in range(Ks_field.shape[0]):
            plt.figure()
            plt.pcolormesh(Ks_field[i_layer,:,:],norm=colors.LogNorm(vmin=1e-5, vmax=.5) )
            plt.colorbar()
            if i_layer in i_sample_z:
                method_ = 'Kriged'
            else:
                method_ = 'Interpolated'
            plt.title('Permeability, %s, layer:%i [m/h]' % (method_,i_layer) )
            plt.savefig( os.path.join(folder_figs,'Ks_real%3.3i_%3.3i.png'% (i_real,i_layer)) )
            plt.close()    
    
    
def generate_a_anom(i_real,settings_setup,settings_run):
  
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    file_a_out = os.path.join(dir_real,'a.sa') 
    
    file_indi = settings_setup['file_indi']
    plot = settings_setup['a_plot']
    folder_figs = os.path.join(dir_real,'figures') 
    dir_in = settings_run['dir_DA']
    mode = 'ml'

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
    if mode == 'rnd':
        rnd_seed = np.random.randint(100)
   

    ### 1) indicator file data
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

    
    ### generate alpha field from indicator data
    field_a_log = np.zeros(data_indi.shape)

    for i1,indicator in enumerate(indi_nr):
        mask = (data_indi == indicator)

        if indicator <= 12: #surface layers, soilgrids
            # use Rosetta: values are given in log10(cm/day) -> convert to log10(m/h) for ParFlow
            field_a_log[mask] = Ros_loga[i1] + np.log10(100) #go from cm-1 (Rosetta) to m-1 (ParFlow)
        elif indicator < 20: # deep subsurface
            # use standard ParFlow values. Take the log10, and use predefined uncertainty
            field_a_log[mask] = indi_loga[i1]
        else: #lake and ocean
            field_a_log[mask] = indi_loga[i1]
    assert (field_a_log==0).sum() == 0
    
    
    ### 2) Add anomaly field
    anom_water = 0 # no anomaly for water/alluvium
    anom_allv = 0 #
    indi_water = 21
    indi_allv = 13
    
    X_train = np.load(os.path.join(dir_in,'a_anom.static.npy'))
    Y_train = np.load(os.path.join(dir_in,'a_anom.param.%3.3i.%3.3i.%3.3i.npy'%(settings_setup['i_date'],settings_setup['i_iter'],i_real) ))
      
    anom_field = generate_anomaly_field(X_train,Y_train,data_indi,mode='ml')
    
    anom_field[data_indi >= indi_water] = anom_water
    anom_field[data_indi == indi_allv] = anom_allv
   
    a_field = 10**(field_a_log + anom_field)
 
    writeSa(file_a_out,a_field)

    if plot:
        i_sample_z = np.unique(X_train[:,0])
        
        if not os.path.exists(folder_figs):
            print('Making folder for verification plots in %s' % folder_figs )
            os.mkdir(folder_figs)

        for i_layer in range(a_field.shape[0]):
            plt.figure()
            plt.pcolormesh(a_field[i_layer,:,:],norm=colors.LogNorm() )
            plt.colorbar()
            if i_layer in i_sample_z:
                method_ = 'Kriged'
            else:
                method_ = 'Interpolated'
            plt.title('Alpha (v. Genuchten), %s, layer:%i [m/h]' % (method_,i_layer) )
            plt.savefig( os.path.join(folder_figs,'a_real%3.3i_%3.3i.png'% (i_real,i_layer)) )
            plt.close()    
            

def generate_n_anom(i_real,settings_setup,settings_run):
  
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    file_n_out = os.path.join(dir_real,'n.sa') 
    
    file_indi = settings_setup['file_indi']
    plot = settings_setup['n_plot']
    folder_figs = os.path.join(dir_real,'figures') 
    dir_in = settings_run['dir_DA']
    mode = 'ml'

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
    if mode == 'rnd':
        rnd_seed = np.random.randint(100)
   

    ### 1) indicator file data
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

    
    ### generate alpha field from indicator data
    field_n_log = np.zeros(data_indi.shape)

    for i1,indicator in enumerate(indi_nr):
        mask = (data_indi == indicator)

        if indicator <= 12: #surface layers, soilgrids
            # use Rosetta: values are given in log10(cm/day) -> convert to log10(m/h) for ParFlow
            field_n_log[mask] = Ros_logn[i1] #no need to go from from cm-1 to m-1: n is unitless
        elif indicator < 20: # deep subsurface
            # use standard ParFlow values. Take the log10, and use predefined uncertainty
            field_n_log[mask] = indi_logn[i1]
        else: #lake and ocean
            field_n_log[mask] = indi_logn[i1]
    assert (field_n_log==0).sum() == 0
    
    
    ### 2) Add anomaly field
    anom_water = 0 # no anomaly for water/alluvium
    anom_allv = 0 #
    indi_water = 21
    indi_allv = 13
    
    X_train = np.load(os.path.join(dir_in,'n_anom.static.npy'))
    Y_train = np.load(os.path.join(dir_in,'n_anom.param.%3.3i.%3.3i.%3.3i.npy'%(settings_setup['i_date'],settings_setup['i_iter'],i_real) ))
      
    anom_field = generate_anomaly_field(X_train,Y_train,data_indi,mode='ml')
    
    anom_field[data_indi >= indi_water] = anom_water
    anom_field[data_indi == indi_allv] = anom_allv
   
    n_field = 10**(field_n_log + anom_field)
 
    writeSa(file_n_out,n_field)

    if plot:
        i_sample_z = np.unique(X_train[:,0])
        
        if not os.path.exists(folder_figs):
            print('Making folder for verification plots in %s' % folder_figs )
            os.mkdir(folder_figs)

        for i_layer in range(n_field.shape[0]):
            plt.figure()
            plt.pcolormesh(n_field[i_layer,:,:],norm=colors.LogNorm() )
            plt.colorbar()
            if i_layer in i_sample_z:
                method_ = 'Kriged'
            else:
                method_ = 'Interpolated'
            plt.title('n (v. Genuchten), %s, layer:%i [m/h]' % (method_,i_layer) )
            plt.savefig( os.path.join(folder_figs,'n_real%3.3i_%3.3i.png'% (i_real,i_layer)) )
            plt.close()  
  

def generate_poros_anom(i_real,settings_setup,settings_run):
  
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    file_poros_out = os.path.join(dir_real,'poros.sa') 
    
    file_indi = settings_setup['file_indi']
    plot = settings_setup['poros_plot']
    folder_figs = os.path.join(dir_real,'figures') 
    dir_in = settings_run['dir_DA']
    mode = 'ml'

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
    if mode == 'rnd':
        rnd_seed = np.random.randint(100)
   

    ### 1) indicator file data
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

    
    ### generate alpha field from indicator data
    field_poros = np.zeros(data_indi.shape)

    for i1,indicator in enumerate(indi_nr):
        mask = (data_indi == indicator)

        if indicator <= 12: #surface layers, soilgrids
            field_poros[mask] = Ros_poros[i1] #
        elif indicator < 20: # deep subsurface
            # use standard ParFlow values. 
            field_poros[mask] = indi_poros[i1]
        else: #lake and ocean
            field_poros[mask] = indi_poros[i1]
    assert (field_poros==0).sum() == 0
    
    
    ### 2) Add anomaly field
    anom_water = 0 # no anomaly for water/alluvium
    anom_allv = 0 #
    indi_water = 21
    indi_allv = 13

    X_train = np.load(os.path.join(dir_in,'poros_anom.static.npy'))
    Y_train = np.load(os.path.join(dir_in,'poros_anom.param.%3.3i.%3.3i.%3.3i.npy'%(settings_setup['i_date'],settings_setup['i_iter'],i_real) ))
      
    anom_field = generate_anomaly_field(X_train,Y_train,data_indi,mode='ml')
    
    anom_field[data_indi >= indi_water] = anom_water
    anom_field[data_indi == indi_allv] = anom_allv
   
    field_poros = field_poros + anom_field
 
    # cap fields just in case
    field_poros[field_poros>.6] = .6
    field_poros[field_poros<.2] = .2

    writeSa(file_poros_out,field_poros)

    if plot:
        i_sample_z = np.unique(X_train[:,0])
        
        if not os.path.exists(folder_figs):
            print('Making folder for verification plots in %s' % folder_figs )
            os.mkdir(folder_figs)

        for i_layer in range(field_poros.shape[0]):
            plt.figure()
            plt.pcolormesh(field_poros[i_layer,:,:],vmin=.3,vmax=.5 )
            plt.colorbar(extend='both')
            if i_layer in i_sample_z:
                method_ = 'Kriged'
            else:
                method_ = 'Interpolated'
            plt.title('porosity, %s, layer:%i [m/h]' % (method_,i_layer) )
            plt.savefig( os.path.join(folder_figs,'poros_real%3.3i_%3.3i.png'% (i_real,i_layer)) )
            plt.close()  
 

def generate_slope_anom(i_real,settings_setup,settings_run):
  
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    file_slopex_in = settings_setup['file_slopex']
    file_slopey_in = settings_setup['file_slopey']
    file_slopex_out = os.path.join(dir_real,'slopex.sa') 
    file_slopey_out = os.path.join(dir_real,'slopey.sa') 

    file_indi = settings_setup['file_indi']
    
    sample_every_xy = settings_setup['slope_sample_xy']

    plot = settings_setup['slope_plot']
    folder_figs = os.path.join(dir_real,'figures') 
    dir_in = settings_run['dir_DA']
    mode = 'ml'

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
    if mode == 'rnd':
        rnd_seed = np.random.randint(100)
   
    ### 1) read slope files, and indicator file (used as mask)
    data_x = readSa(file_slopex_in)
    data_y = readSa(file_slopey_in)
    data_indi = readSa(file_indi)
    
    
    ### 2) Add anomaly field
    anom_water = 0 # no anomaly for water/alluvium
    indi_water = 21

    X_train = np.load(os.path.join(dir_in,'slope_anom.static.npy'))
    Y_train = np.load(os.path.join(dir_in,'slope_anom.param.%3.3i.%3.3i.%3.3i.npy'%(settings_setup['i_date'],settings_setup['i_iter'],i_real) ))
      
    anom_field = generate_anomaly_field(X_train,Y_train,data_indi,mode='ml',shape_out=data_x.shape)
    anom_field[data_indi[-2:-1,:,:] >= indi_water] = anom_water #-2:-1 is used to keep the first dimension 1 
   
    data_x = data_x*(10**(anom_field)) # anomaly field: 10**-1 -> 10**0 -> 10**1: 0.1 -> 1 -> 10 (multiplicative factor)
    data_y = data_y*(10**(anom_field))
    
    writeSa(file_slopex_out,data_x)
    writeSa(file_slopey_out,data_y)

    if plot:
        if not os.path.exists(folder_figs):
            print('Making folder for verification plots in %s' % folder_figs )
            os.mkdir(folder_figs)

        plt.figure()
        plt.pcolormesh(data_x[0,:,:] )
        plt.colorbar(extend='both')
        plt.title('slope (x)')
        plt.savefig( os.path.join(folder_figs,'slopex_real%3.3i.png'% (i_real)) )
        plt.close()  

        plt.figure()
        plt.pcolormesh(data_y[0,:,:] )
        plt.colorbar(extend='both')
        plt.title('slope (y)')
        plt.savefig( os.path.join(folder_figs,'slopey_real%3.3i.png'% (i_real)) )
        plt.close()  
        
    
if __name__ == '__main__':
    
    file_indi = '/p/project/cjibg36/kaandorp2/TSMP_patched/tsmp_cordex_111x108/input_pf/EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_111x108_INDICATOR_regridded_rescaled_SoilGrids250-v2017_BGR3_alv.sa'
    file_Ks_out = '/p/project/cjibg36/kaandorp2/TSMP_patched/tsmp_cordex_111x108_KsKriged/input_pf/Ks_kriged.sa'
    mode = 'ml' #rnd or ml
    realization = 0
    plot=False
    folder_figs='figures'
    
    generate_Ks(file_indi,file_Ks_out,mode,realization=realization,plot=plot,folder_figs=folder_figs)