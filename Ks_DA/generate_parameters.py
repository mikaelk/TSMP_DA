import numpy as np
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import os
import xarray as xr

extra_path = '/p/project/cjibg36/kaandorp2/Git/SLOTH' 
if extra_path not in sys.path:
    sys.path.append(extra_path)

from sloth.IO import readSa, writeSa


def generate_anomaly_field(X_train,Y_train,data_indi,mode='ml',shape_out=None):
    """
    Generate anomaly field using Gaussian Process Regression
    X_train: [X,Y,Z] coordinates
    Y_train: corresponding response
    data_indi: indicator field, used to mask ocean
    mode: most likely (ml) or additional random noise based on Kriged function (rnd, don't use this)
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

    kernel = 1.0 * Matern(length_scale=30,nu=1.5,length_scale_bounds='fixed')

    indi_dz = 2*np.array([9.0,7.5,5.0,5.0,2.0,0.5,0.35,0.25,0.15,0.10,0.065,0.035,0.025,0.015,0.01])
    indi_depths = np.flip(np.cumsum(np.flip(indi_dz)))
    ### /parameters

    # indices of layers to be sampled by Kriging
    i_sample_z = np.unique(X_train[:,0])
    # fit the Gaussian Process Regressors
    gprs = fit_gprs(i_sample_z,X_train,Y_train)


    X = np.meshgrid(np.arange(0,anom_field.shape[1]),
                    np.arange(0,anom_field.shape[2]),indexing='ij')

    i_lower_old = np.inf

    for i_layer in range(anom_field.shape[0]):
        print('Generating anomaly field for layer %i' % i_layer,flush=True)
        mask_land = (data_indi[i_layer,:] < 20)
        X_test = np.array([X[0][mask_land],
                        X[1][mask_land]]).T


        # layer is sampled -> Kriging
        if i_layer in i_sample_z:
            if mode == 'ml':
                anom_ = gprs[i_layer].predict(X_test)
            elif mode == 'rnd':
                anom_ = gprs[i_layer].sample_y(X_test,random_state=rnd_seed)[:,0]
            else:
                raise RuntimeError('Mode should be ml (most likely) or rnd (random)')
            anom_field[i_layer,mask_land] = anom_

        # layer is not sampled -> interpolate linearly
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
    mode = 'rnd'
    length_scale = settings_gen['texture_anom_l']

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

    ### 2) Add anomaly field
    X_train = np.load(os.path.join(dir_in,'sandfrac_anom.static.npy'))
    Y_train = np.load(os.path.join(dir_in,'sandfrac_anom.param.%3.3i.%3.3i.%3.3i.npy'%(settings_gen['i_date'],settings_gen['i_iter'],i_real) ))

    anom_field = generate_anomaly_field(X_train,Y_train,data_texture,mode='ml',
                                        shape_out=data_texture.shape,mask_land=mask_land,length_scale=length_scale)
    anom_field[:,~mask_land]=0

    data_texture = data_texture*(10**(anom_field)) # anomaly field: 10**-1 -> 10**0 -> 10**1: 0.1 -> 1 -> 10 (multiplicative factor)

    max_sand = 99. 
    data_texture[data_texture>max_sand] = max_sand

    data['PCT_SAND'] = (['nlevsoi','lsmlat','lsmlon'],data_texture)

    data.to_netcdf(file_out)

    if plot:
        if not os.path.exists(folder_figs):
            print('Making folder for verification plots in %s' % folder_figs )
            os.mkdir(folder_figs)

        plt.figure()
        plt.pcolormesh(data_texture[0,:,:] )
        plt.colorbar(extend='both')
        plt.title('Sand fraction (upper layer)')
        plt.savefig( os.path.join(folder_figs,'sandfrac_real%3.3i.png'% (i_real)) )
        plt.close()  
    
    
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


if __name__ == '__main__':
    
    file_indi = '/p/project/cjibg36/kaandorp2/TSMP_patched/tsmp_cordex_111x108/input_pf/EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_111x108_INDICATOR_regridded_rescaled_SoilGrids250-v2017_BGR3_alv.sa'
    file_Ks_out = '/p/project/cjibg36/kaandorp2/TSMP_patched/tsmp_cordex_111x108_KsKriged/input_pf/Ks_kriged.sa'
    mode = 'ml' #rnd or ml
    realization = 0
    plot=False
    folder_figs='figures'
    
    generate_Ks(file_indi,file_Ks_out,mode,realization=realization,plot=plot,folder_figs=folder_figs)