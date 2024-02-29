import numpy as np
import os
import re
import shutil
from datetime import datetime, timedelta
import pandas as pd
from glob import glob
import xarray as xr
import netCDF4

from run_realization_v2 import setup_submit_wait
from DA_operators import operator_clm_SMAP, operator_clm_FLX

import multiprocessing as mp

from itertools import repeat
from scipy import sparse

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import time
import sys
import pickle

folder_results = '/p/scratch/cjibg36/kaandorp2/TSMP_results/eTSMP/DA_eCLM_cordex_444x432_v14_1y_iter5'
if folder_results not in sys.path:
    sys.path.insert(0,os.path.join(folder_results,'settings'))
from settings_copy import settings_run,settings_clm,settings_pfl,settings_sbatch,settings_DA,settings_gen,date_results_binned,freq_output,date_range_noleap

dir_figs = os.path.join(folder_results,'figures/02_gridded_SMAP_mismatch')
if not os.path.exists(dir_figs):
    print('Creating folder to store gridded SMAP mismatch: %s' % (dir_figs) )
    os.makedirs(dir_figs)
    
pickle_filename = os.path.join(dir_figs,'gridded_mismatch.pickle')
pickle_filename_tmp = os.path.join(dir_figs,'gridded_mismatchd_tmp.pickle')

def haversine_distance_2d(loc1, loc_array):
    """
    Calculate the Haversine distance between a point and an array of points on the Earth
    given their latitude and longitude in decimal degrees.

    Parameters:
    - loc1: Tuple containing the latitude and longitude of the first point (in decimal degrees).
    - loc_array: 2D arrays with latitudes and longitudes of points (in decimal degrees).

    Returns:
    - Array of distances between loc1 and each point in loc_array (in kilometers).
    """
    if np.isnan(loc1[0]) and np.isnan(loc1[1]):
        distances = np.zeros(loc_array.shape[0])
    else:
        # Radius of the Earth in kilometers
        R = 6371.0

        # Convert decimal degrees to radians
        lat1_rad, lon1_rad = np.radians(loc1)
        lat2_rad, lon2_rad = np.radians(loc_array[0,:,:]), np.radians(loc_array[1,:,:])

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distances = R * c
    return distances

def haversine_distance_vec(loc1_array, loc_array):
    """
    Vectorized version of the Haversine distance
    Calculate the Haversine distance between an array of points and another array of points on the Earth
    given their latitude and longitude in decimal degrees.

    Parameters:
    - loc1_array: Array of tuples, each containing the latitude and longitude of a point (in decimal degrees).
    - loc_array: Array of tuples, each containing the latitude and longitude of a point (in decimal degrees).

    Returns:
    - 2D array of distances between each point in loc1_array and each point in loc_array (in kilometers).
    """
    if np.isnan(loc1_array).all(axis=1).any():
        distances = np.zeros((len(loc1_array), len(loc_array)))
    else:
        # Radius of the Earth in kilometers
        R = 6371.0

        # Convert decimal degrees to radians
        lat1_rad, lon1_rad = np.radians(np.array(loc1_array).T)
        lat2_rad, lon2_rad = np.radians(np.array(loc_array).T)

        # Haversine formula
        dlat = lat2_rad[:, np.newaxis] - lat1_rad
        dlon = lon2_rad[:, np.newaxis] - lon1_rad

        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad[:, np.newaxis]) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distances = R * c

    return distances

def realize_parameters(i_real,settings_gen,settings_run,init=True,run_prior=False):
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    local_state = np.random.RandomState() #required for parallel processes in python
    dir_DA = settings_run['dir_DA']

    if not os.path.exists(dir_real):
        print('Creating parameter realizations for ensemble member %i' % i_real)
        print('Creating folder for realization %i: %s' % (i_real,dir_real), flush=True )
        os.mkdir(dir_real)
        time.sleep(1)
        
        if init:
            print('Initializing parameters from prior parameter settings')
            i_iter_ = 0 #prior parameters are always read from the initial iteration file
            # Read parameter values + std, generate parameter realizations (i_real)
            for p_name, p_fn_gen in zip(settings_gen['param_names'],settings_gen['param_gen']):
                p_values = np.load(os.path.join(dir_DA,'%s.param.%3.3i.%3.3i.prior.npy'% (p_name,settings_gen['i_date'],i_iter_) ))
                p_mean = p_values[:,0]
                p_sigma = p_values[:,1]
                
                # ensemble member 0: most likely parameter values are used
                if i_real == 0 or run_prior:
                    p_real = p_mean.copy()
                else:
                    p_real = local_state.normal(p_mean,p_sigma)
                np.save(os.path.join(dir_DA,'%s.param.%3.3i.%3.3i.%3.3i'%(p_name,settings_gen['i_date'],settings_gen['i_iter'],i_real)),p_real)
                p_fn_gen(i_real,settings_gen,settings_run)
        else:
            print('Updating parameters from DA analysis')
            for p_name, p_fn_gen in zip(settings_gen['param_names'],settings_gen['param_gen']):
                print('Debug: realize_parameters, i_real %i, %s, %s' % (i_real,p_name, p_fn_gen) )
                p_fn_gen(i_real,settings_gen,settings_run)
            
def worker_realize_parameters(*args, **kwargs):
    try:
        realize_parameters(*args, **kwargs)
    except Exception as e:
        print(f"Exception in worker: {e}")

            
def read_parameters(n_ensemble,settings_gen,settings_run):
    # read parameter values of the different ensemble members into an array
    param_names = []
    param_latlon = np.array([])
    param_r_loc = np.array([])
    param_lengths_old = []
    for i1 in np.arange(n_ensemble):
        param_tmp = np.array([])
        
        if i1 == 0:
            for i2,p_name in enumerate(settings_gen['param_names']):
                param_ = np.load(os.path.join(settings_run['dir_DA'],'%s.param.%3.3i.%3.3i.%3.3i.npy'% (p_name,settings_gen['i_date'],settings_gen['i_iter'],i1+1) ))
                # settings_gen['param_length'][p_name] = len(param_)
                param_lengths_old.append(len(param_))
                param_tmp = np.append(param_tmp,param_)
                param_names.extend([p_name + '_%i'%int_ for int_ in np.arange(len(param_))])
                param_r_loc = np.append(param_r_loc,settings_gen['param_r_loc'][i2]*np.ones(len(param_)))
                file_latlon = os.path.join(settings_run['dir_DA'],'%s.latlon.npy'% (p_name))
                if os.path.exists(file_latlon):
                    if len(param_latlon) == 0:
                        param_latlon = np.load(file_latlon)
                    else:
                        param_latlon = np.vstack((param_latlon,np.load(file_latlon)))
                else:
                    if len(param_latlon) == 0:
                        param_latlon = np.nan*np.zeros([len(param_),2])
                    else:
                        param_latlon = np.vstack((param_latlon,np.nan*np.zeros([len(param_),2])))
            param_all = param_tmp.copy()
            
        else:
            param_lengths = []
            for i2,p_name in enumerate(settings_gen['param_names']):
                param_ = np.load(os.path.join(settings_run['dir_DA'],'%s.param.%3.3i.%3.3i.%3.3i.npy'% (p_name,settings_gen['i_date'],settings_gen['i_iter'],i1+1) ))
                param_lengths.append(len(param_))
                param_tmp = np.append(param_tmp,param_)        
            param_all = np.vstack((param_all,param_tmp))
            
            if param_lengths != param_lengths_old:
                raise RuntimeError('parameter lengths not equal\n%s\n%s' % (param_lengths,param_lengths_old))
                
    return param_all.T,param_names,param_latlon,param_r_loc


def write_parameters(parameters,settings_gen,settings_run):
    dir_DA = settings_run['dir_DA']
    for i_real in range(parameters.shape[1]):
        i_start = 0
        for p_name in settings_gen['param_names']:
            i_end = i_start + settings_gen['param_length'][p_name] 
            param_ = parameters[i_start:i_end,i_real]
            np.save(os.path.join(dir_DA,'%s.param.%3.3i.%3.3i.%3.3i'%(p_name,settings_gen['i_date'],settings_gen['i_iter']+1,i_real+1)),param_)
            i_start = i_end
            
    # write mean parameter values to member 0
    i_start = 0
    for p_name in settings_gen['param_names']:
        i_end = i_start + settings_gen['param_length'][p_name] 
        param_ = parameters[i_start:i_end,:].mean(axis=1)
        np.save(os.path.join(dir_DA,'%s.param.%3.3i.%3.3i.%3.3i'%(p_name,settings_gen['i_date'],settings_gen['i_iter']+1,0)),param_)
        i_start = i_end   
        
def change_setting(filename, key, new_value):
    # Escape special characters in the key
    escaped_key = re.escape(key)

    # Define the pattern to match
    pattern = re.compile(r"('{}'\s*:\s*)(.+?)(?=[,}}])".format(escaped_key))

    # Read the content of the file
    with open(filename, 'r') as file:
        content = file.read()

    # Use the pattern to find and replace the matched value
    content = pattern.sub(r"\g<1>{}".format(new_value), content)

    # Write the updated content back to the file
    with open(filename, 'w') as file:
        file.write(content)
        
def check_for_success(dir_iter,dir_DA,dir_settings,date_results_iter,n_ensemble):
        
    date_start_sim = date_results_iter[-1][0]
    date_end_sim = date_results_iter[-1][-1]
    str_date = str(date_start_sim.date()).replace('-','') + '-' + str(date_end_sim.date()).replace('-','')
    all_success = False
    reread_required = False

    ## Following function should be called iteratively until np.all(flag_success) = True
    while not all_success:
        reread_required = True
        i_source = n_ensemble #which directory to move in case one of the runs failed

        flag_success = np.zeros(n_ensemble,dtype=bool)
        for i1 in range(1,n_ensemble+2):
            restart_file = glob(os.path.join(dir_iter,'R%3.3i/run_%s/*.clm2.r.*.nc'%(i1,str_date)))
            print('Restart file: %i, %s' %(i1,restart_file) )
            if len(restart_file) > 0:
                flag_success[i1-1] = True


        # if the last run failed, remove it
        while flag_success[-1] == False:
            # remove last folder
            shutil.rmtree(os.path.join(dir_iter,'R%3.3i'%(i_source)), ignore_errors = True) 
            # remove last index flag
            flag_success = np.delete(flag_success,-1)
            i_source -= 1
            n_ensemble -= 1

        if not np.all(flag_success):
            i_dest = np.where(~flag_success)[0][0]+1 

            print('-----------------Important!!!!!-----------------')
            print('Moving R%3.3i to R%3.3i (failed run)' %(i_source,i_dest) )
            print('-----------------Important!!!!!-----------------')

            paramfiles_source = sorted(glob(os.path.join(dir_DA,'*.%3.3i.npy'%i_source)))
            paramfiles_source_tmp = sorted([file_ + '_old' for file_ in paramfiles_source])
            paramfiles_dest = sorted(glob(os.path.join(dir_DA,'*.%3.3i.npy'%i_dest)))
            paramfiles_dest_tmp = sorted([file_ + '_old' for file_ in paramfiles_dest])

            shutil.move(os.path.join(dir_iter,'R%3.3i'%i_dest),os.path.join(dir_iter,'R%3.3i_old'%i_dest) )
            shutil.move(os.path.join(dir_iter,'R%3.3i'%i_source),os.path.join(dir_iter,'R%3.3i'%i_dest) )

            for file_src_in,file_src_out,file_dest_in,file_dest_out in zip(paramfiles_source,paramfiles_source_tmp,paramfiles_dest,paramfiles_dest_tmp):
                shutil.move(file_dest_in,file_dest_out) #move failed parameter file (e.g. 13) to 13_old
                shutil.copy(file_src_in,file_src_out) #copy the successfull parameter file (e.g. 16) to 16_old
                shutil.move(file_src_in,file_dest_in) #move the successfull paramfile (e.g. 16) to 13

            n_ensemble -=1

        else:
            all_success = True
            
    if not reread_required:
        print('Simulations were all successfull, all restart files available')
    else:
        change_setting(os.path.join(dir_settings,'settings.py'),'n_ensemble',n_ensemble)
        
    return n_ensemble, reread_required


def mask_observations(data_names,data_measured,data_var,data_latlon,data_nselect,data_mask,factor_inflate=1.):
    data_indices = {}
    i_start = 0
    i_end = np.inf
    
    # print(factor_inflate)
    n_vars = len(data_names)
    if type(factor_inflate) == float:
        var_inflate = {var_ : factor_inflate for var_ in data_names}
    elif type(factor_inflate) == dict:
        var_inflate = factor_inflate.copy()
    else:
        raise RuntimeError('type should be float or dict')
        
    for i1,var_ in enumerate(data_names):
        # print(var_,data_var[var_],var_inflate[var_])
        data_var[var_] *= var_inflate[var_]
        
        n_select = data_nselect[var_]
        if len(data_measured[var_]) < n_select:
            data_mask[var_] = np.ones(len(data_measured[var_]),dtype=bool)
            n_select = len(data_measured[var_])
        else:
            frac_select = min((n_select / len(data_measured[var_])),.99)
            data_mask[var_] = np.random.choice([0,1],size=len(data_measured[var_]),p=[1-frac_select,frac_select]).astype(bool) 
            n_select = int(data_mask[var_].sum())
        i_end = i_start + n_select
        data_indices[var_] = [i_start,i_end]

        if i1 == 0:
            data_measured_masked = data_measured[var_][data_mask[var_]].copy()
            data_latlon_masked = data_latlon[var_][data_mask[var_]].copy()
            if type(data_var[var_]) == float:
                data_var_masked = data_var[var_]*np.ones(n_select)
            else:
                data_var_masked = data_var[var_][data_mask[var_]].copy()
        else:
            data_measured_masked = np.append(data_measured_masked,data_measured[var_][data_mask[var_]])
            data_latlon_masked = np.vstack((data_latlon_masked,data_latlon[var_][data_mask[var_]]))
            if type(data_var[var_]) == float:
                data_var_masked = np.append(data_var_masked,data_var[var_]*np.ones(n_select))
            else:
                data_var_masked = np.append(data_var_masked,data_var[var_][data_mask[var_]])

        i_start = i_end
        print('Thinned out %s observations: %i -> %i' % (var_,len(data_measured[var_]),n_select))

    n_data = i_end
    # data_var_masked*=factor_inflate
    return data_mask,data_indices,n_data,data_measured_masked,data_var_masked,data_latlon_masked



def plot_prior_post(param_f,param_a,param_names_all,i_iter,dir_figs=os.path.join('.','params') ):
    if not os.path.exists(dir_figs):
        print('Creating folder to store parameter update figures: %s' % (dir_figs) )
        os.mkdir(dir_figs)
    c = 0
    c2 = 0
    n_param_max = 16*100 #max 100 plots
    n_param = min(len(param_f),n_param_max)
    n_figs = np.ceil(n_param/16).astype(int)

    for i_ in np.arange(n_param):
        if i_ % 16 == 0:
            c = 0
            fig,axes=plt.subplots(4,4,figsize=(8,7))
        row_ = c//4
        col_ = c%4

        axes[row_,col_].plot(param_f[i_,:],np.zeros(param_f[i_,:].shape),'ko')
        axes[row_,col_].plot(param_a[i_,:],np.zeros(param_a[i_,:].shape),'rx')
        axes[row_,col_].set_title(param_names_all[i_])
        c += 1
        if c == 16 or i_ == n_param-1:
            fig.suptitle('Iter %i -> %i' %(i_iter,i_iter+1))
            fig.tight_layout()
            fig.savefig(os.path.join(dir_figs,'params_i%3.3i_%3.3i.png'%(i_iter,c2)) )
            c2 += 1

            
def update_step_ESMDA(param_f,data_f,data_measured,data_var,alpha,i_iter):
    print('Calculating KG and performing parameter update...')
    assert data_f.shape[0] == len(data_measured)
    n_data_ = data_f.shape[0]
    n_ensemble = data_f.shape[1]
    n_param = param_f.shape[0]
    
    # 3) construct covariance matrices based on ensemble of parameters and results (data)
    # C_D = data_var['SMAP']*sparse.eye(n_data) 
    C_D = sparse.diags(data_var)
    C_MD = np.zeros([n_param,n_data_],dtype=np.float32)
    C_DD = np.zeros([n_data_,n_data_],dtype=np.float32)
    param_mean = param_f.mean(axis=1)
    data_mean = data_f.mean(axis=1)        
    param_delta = np.zeros([n_param,n_ensemble])
    data_delta = np.zeros([n_data_,n_ensemble])
    for i2 in range(n_ensemble):
        param_delta[:,i2] = param_f[:,i2] - param_mean
        data_delta[:,i2] = data_f[:,i2] - data_mean

        C_MD += np.outer(param_delta[:,i2],data_delta[:,i2])
        C_DD += np.outer(data_delta[:,i2],data_delta[:,i2])
    C_MD /= (n_ensemble - 1)
    C_DD /= (n_ensemble - 1)

    # Kalman Gain matrix:
    KG = np.dot(C_MD,np.linalg.inv(C_DD + alpha[i_iter]*C_D)) 

    # 4) update the parameters
    param_a = np.zeros([n_param, n_ensemble])
    mean_mismatch_new = 0
    for i_real in range(n_ensemble):

        z_d = np.random.normal(0,1,n_data_)
        data_perturbed = data_measured + np.sqrt(alpha[i_iter])*np.sqrt(C_D.diagonal())*z_d

        mismatch = data_perturbed - data_f[:,i_real]

        mean_mismatch_new += np.sum(mismatch**2)

        # forecast -> analysis
        param_a[:,i_real] = param_f[:,i_real] + np.dot(KG,mismatch)

    mean_mismatch_new /= n_ensemble
    
    return param_a, mean_mismatch_new


def update_step_ESMDA_loc(mat_M,mat_D,data_measured,data_var,alpha,i_iter,n_iter,
                          param_latlon=None,param_r_loc=None,data_latlon=None,ksi=.99,
                          dzeta_global=1.,dir_settings='.',factor_inflate_prior=1.):
    """
    Optimized version for many observations
    Possibility to include localisation
    
    Based on appendix of Emerick (2016), j. of Petroleum Science and Engineering 
    doi.org/10.1016/j.petrol.2016.01.029
    """

    def calculate_alphas(lambda_Wd_,n_iter):
        """
        Based on Rafiee and Reynolds, 2017 (Hankes regularization condition)
        Calculate set of inflation factors
        """
        if n_iter == 1:
            alphas = [1]
        else:
            try:
                from scipy.optimize import minimize_scalar
                def f1(gamma,alpha_1,N_a):
                        sum_ = 0
                        for i in range(N_a):
                            k= i+1
                            if ( (gamma**(k-1)) * alpha_1) == 0:
                                print(gamma,k,alpha_1)
                            sum_ += (1/ ( (gamma**(k-1)) * alpha_1) )

                        return (sum_ - 1)**2

                rho = .5
                alpha_1 = max((rho/(1-rho))*lambda_Wd_.mean()**2,n_iter)

                res = minimize_scalar(f1,bounds=(0.001,0.999),bracket=(0.001,0.999),args=(alpha_1,n_iter))
                gamma = res.x

                alphas = [(gamma**k)*alpha_1 for k in np.arange(n_iter)]
                print('Inflation factors (alpha) calculated:',alphas)
            except:
                print('Inflation factor (alpha) calculation failed, falling back to alpha=n_iter')
                alphas = [n_iter for k in np.arange(n_iter)]

            assert( np.sum(1/np.array(alphas))-1 < 1e-5)

        return alphas

    print('Calculating KG and performing parameter update...', flush=True)
    assert mat_D.shape[0] == len(data_measured)

    n_data_ = mat_D.shape[0]
    n_ensemble = mat_D.shape[1]
    n_param = mat_M.shape[0]

    C_D = sparse.diags(data_var)

    del_M = (1/np.sqrt(n_ensemble-1))*(mat_M - mat_M.mean(axis=1)[:,np.newaxis])
    del_D = (1/np.sqrt(n_ensemble-1))*(mat_D - mat_D.mean(axis=1)[:,np.newaxis])

    del_D *= factor_inflate_prior
    
    S = sparse.diags(C_D.diagonal()**(1/2))
    C_Dh = 1.*sparse.eye(n_data_)
    Sinv = sparse.diags(1/S.diagonal()) #inverse of diagonal matrix: simply the reciprocal

    Ud, lambda_Wd, Vdt = np.linalg.svd(Sinv.dot(del_D), full_matrices=False)
    assert(np.all(lambda_Wd[:-1] > lambda_Wd[1:])) #assert that the eigenvalues are sorted

    if alpha is None:
        print('Calculating inflation factors alpha...', flush=True)
        alpha = calculate_alphas(lambda_Wd,n_iter)
        print(alpha, flush=True)
        change_setting(os.path.join(dir_settings,'settings.py'),'alpha',alpha)
        
    cumsum_wr = np.cumsum(lambda_Wd) / np.sum(lambda_Wd)
    Nr = max(len(lambda_Wd)//2, np.where(cumsum_wr<=ksi)[0][-1]) #take Nr most important singular values, retain at least half the original matrix size just in case

    Ur = Ud[:,0:Nr]
    Wr = sparse.diags(lambda_Wd[0:Nr])
    Vrt = Vdt[0:Nr,0:Nr]
    Ir = sparse.eye(Nr)
    Wrinv = sparse.diags(1/Wr.diagonal()) 

    mat_R = alpha[i_iter]*(Wrinv @ Ur.T @ C_Dh @ Ur @ Wrinv)

    Zr, lambda_Hr, Zrt = np.linalg.svd(mat_R, full_matrices=False)
    Hr = sparse.diags(lambda_Hr)

    mat_X = Sinv @ Ur @ Wrinv @ Zr
    mat_L = Ir + Hr
    mat_Linv = sparse.diags(1/mat_L.diagonal()) 

    mat_X1 = mat_Linv @ mat_X.T
    mat_X2 = del_D.T @ mat_X
    mat_X3 = mat_X2 @ mat_X1

    # perturb observations, at the same time calculate the mismatch of the current forecast
    mat_Dobs = np.zeros(mat_D.shape)
    mean_mismatch_new = 0
    for i_real in range(n_ensemble):
        z_d = np.random.normal(0,1,n_data_)
        mat_Dobs[:,i_real] = data_measured + np.sqrt(alpha[i_iter])*np.sqrt(C_D.diagonal())*z_d
        mismatch = mat_Dobs[:,i_real] - mat_D[:,i_real]
        mean_mismatch_new += np.sum(mismatch**2)
    mean_mismatch_new /= n_ensemble
    
    sum_d_localized = 0
    sum_d_global = 0
    n_param_localized_tot = 0
    # calculate updated (analysis) parameters
    param_a = np.zeros(mat_M.shape)
    for i in range(n_param):
        if np.isnan(param_r_loc[i]):
            rho_i = dzeta_global*np.ones(n_data_) #no localisation
            # rho_i = np.ones(n_data_) #no localisation
            sum_d_global += n_data_
        else:
            r_loc = param_r_loc[i]
            # localisation using the Gaspari-Cohn localisation function:
            rho_i = GC(haversine_distance(param_latlon[i,:],data_latlon),r_loc)
            sum_d_localized += rho_i.sum()
            n_param_localized_tot += 1
            
        K_i = del_M[i,:]@mat_X3
        K_rho_i = K_i * rho_i
        mat_X4 = K_rho_i @ (mat_Dobs - mat_D)
        param_a[i,:] = mat_M[i,:] + mat_X4
    print('Observation statistics:')
    print('Tapering global: %f' % dzeta_global)
    print('Mean observations per localized parameter: %f' % (sum_d_localized/n_param_localized_tot) )
    print('Mean ratio localized/global: %f' % ((sum_d_localized/n_param_localized_tot)/n_data_) )
        
    print('Misc.:')
    print('n_param_global*n_data: %f' % sum_d_global)
    print('n_param_localized*n_data_effective: %f' % sum_d_localized)
    print('Total ratio localized/global: %f' %(sum_d_localized/sum_d_global) )
    
    return param_a, mean_mismatch_new, alpha

def GC(r, c):
    #Gaspari-Cohn localization function
    abs_r = np.abs(r)
    if np.isnan(c):
        result = np.ones_like(abs_r, dtype=float)
    else:
        condition1 = (0 <= abs_r) & (abs_r <= c)
        condition2 = (c <= abs_r) & (abs_r <= 2 * c)

        result = np.zeros_like(abs_r, dtype=float)

        result[condition1] = -1/4 * (abs_r[condition1] / c) ** 5 + 1/2 * (abs_r[condition1] / c) ** 4 + 5/8 * (abs_r[condition1] / c) ** 3 - \
                            5/3 * (abs_r[condition1] / c) ** 2 + 1
        result[condition2] = 1/12 * (abs_r[condition2] / c) ** 5 - 1/2 * (abs_r[condition2] / c) ** 4 + 5/8 * (abs_r[condition2] / c) ** 3 + \
                            5/3 * (abs_r[condition2] / c) ** 2 - 5 * (abs_r[condition2] / c) + 4 - 2/3 * (c / abs_r[condition2])

    return result

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


# which data to assimilate: 
data_names = settings_DA['data_names']
# If uncertainties are assumed constant, prescribe:
data_var = {'SMAP':0.04**2,
            'FLX':None}
# possibility to only select a limited amount of observations using masks
data_nselect = settings_DA['n_data_max']
data_mask = {'SMAP':None, #initialize dict
             'FLX':None}
plot_members_SMAP = 0 #set to int (iteration for which to plot members), or to False/True
plot_members_FLX = True

# prescribe_alpha = settings_DA['prescribe_alpha']
alpha = settings_DA['alpha']
factor_inflate = settings_DA['factor_inflate']
factor_inflate_prior = settings_DA['factor_inflate_prior']
ksi=settings_DA['cutoff_svd']

### Unpack some of the settings into variables
# Functions that are run to initialize the parameters to be assimilated. 
# E.g. for spatial parameter fields, initialize the static fields (x,y,z) locations and the prior/uncertainty estimates
param_setup = settings_DA['param_setup'] 
# Functions that are run to generate realizations of parameters/state variables
param_gen   = settings_DA['param_gen']
# Define parameter names; parameters values are stored in (%s.param.npy % param_name) files
param_names = settings_DA['param_names']

n_parallel = settings_DA['n_parallel']
n_parallel_setup = settings_DA['n_parallel_setup']
n_ensemble = settings_DA['n_ensemble']
n_iter = settings_DA['n_iter']
dir_setup = settings_run['dir_setup']
dir_template = settings_run['dir_template']



dir_DA = os.path.join(dir_setup,'input_DA')
settings_run['dir_DA'] = dir_DA
if not os.path.exists(dir_DA):
    print('Creating folder to store DA information: %s' % (dir_DA) )
    os.mkdir(dir_DA)

    # setup parameters: prior/uncertainties, + static properties, lon/lat locations based on the settings if necessary
    for fn in param_setup:
        fn(settings_gen,settings_run)

# Read parameter length and put in dictionary here
for param_ in param_names:
    settings_gen['param_length'][param_] = np.load(os.path.join(dir_DA,'%s.param.000.000.prior.npy' % param_) ).shape[0]

#%% ----------- DA loop -----------


#%% ----------- date loop -----------    
# this comes in the date loop, e.g. perform the smoother over a period over 1 year:
i_date = 0
date_results_iter = date_results_binned[i_date].copy()
date_start_sim = date_results_binned[i_date][0][0]#datetime(2019,1,2,12,0,0)
date_end_sim = date_results_binned[i_date][-1][-1]#datetime(2019,12,31,12,0,0)

# add spinup if necessary:
if settings_run['ndays_spinup'] is not None:
    date_results_iter.insert(0,list(date_range_noleap(date_start_sim-timedelta(days=settings_run['ndays_spinup']),date_start_sim,periods=2)))


str_date = str(date_start_sim.date()).replace('-','') + '-' + str(date_end_sim.date()).replace('-','')
dir_date = os.path.join(dir_setup,str_date)
if not os.path.exists(dir_date):
    print('Creating folder for dates %s: %s' % (str_date,dir_date) )
    os.mkdir(dir_date)

## TEMP
# date_DA_start = date_end_sim - timedelta(days=30) # spinup, only assimilate last 30 days
# date_DA_start = datetime(2019,1,1,12,0,0)

mismatch_iter = [0]
i_iter = 5
str_iter = 'i%3.3i' % i_iter
dir_iter = os.path.join(dir_date,str_iter)


if settings_run['ndays_validation'] is not None:
    date_validation_start = date_results_iter[-1][-1]
    date_results_iter.append(list(date_range_noleap(date_validation_start,
                                                    date_validation_start+timedelta(days=settings_run['ndays_validation']),freq=freq_output)))
    print('Last iteration: running most likely parameter values and the OL run, including validation timespan: %s-%s'
      %(str(date_end_sim),str(date_end_sim+timedelta(days=settings_run['ndays_validation']))) )
else:
    print('Last iteration: running most likely parameter values and the OL run')

settings_run['dir_iter'] = dir_iter
settings_gen['i_date'] = i_date
settings_gen['i_iter'] = i_iter
settings_gen['param_gen'] = param_gen
settings_gen['param_names'] = param_names
settings_clm['vars_dump'] = settings_clm['vars_dump_val']


operator = {}
data_measured = {}
operator['SMAP'] = operator_clm_SMAP(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_SMAP'],ignore_rivers=False)
_ = operator['SMAP'].get_measurements(date_results_iter,date_DA_start=date_start_sim)
# operator['FLX'] = operator_clm_FLX(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_FLX'],ignore_rivers=False)
# _ = operator['FLX'].get_measurements(date_results_iter,date_DA_start=date_start_sim)

loc_array_2d = np.array([operator['SMAP'].grid_TSMP.lat_centre.values,operator['SMAP'].grid_TSMP.lon_centre.values])

results = {}


for i_real in [0,63]:
    results[i_real] = {}
    _ = operator['SMAP'].interpolate_model_results(i_real,settings_run,indices_z=[0,1],var='SOILLIQ')

    daily_diff = np.nan*np.zeros([len(operator['SMAP'].lats_out.keys()),operator['SMAP'].grid_TSMP.lat_centre.shape[0],operator['SMAP'].grid_TSMP.lat_centre.shape[1]])
    n_obs = np.zeros([len(operator['SMAP'].lats_out.keys()),operator['SMAP'].grid_TSMP.lat_centre.shape[0],operator['SMAP'].grid_TSMP.lat_centre.shape[1]])

    for i_date,date_ in enumerate(list(operator['SMAP'].sm_out.keys())):
        # i_date = 0
        # date_ = list(operator['SMAP'].sm_out.keys())[0]
        print(date_)

        diff = operator['SMAP'].sm_out[date_] - operator['SMAP'].data_TSMP_i[date_] 

        for i1,(lat_,lon_,diff_) in enumerate(zip(operator['SMAP'].lats_out[date_],operator['SMAP'].lons_out[date_],diff)):
            # lat_ = operator['SMAP'].lats_out[date_][0]
            # lon_ = operator['SMAP'].lons_out[date_][0]
            dist = haversine_distance_2d(np.array([lat_,lon_]).T,loc_array_2d)

            i_min = np.unravel_index(dist.argmin(), dist.shape)

            daily_diff[i_date,i_min[0],i_min[1]] = diff_
            n_obs[i_date,i_min[0],i_min[1]] += 1

            if i1%5000 == 0:
                print('%i/%i' % (i1,len(operator['SMAP'].lats_out[date_])))

    results[i_real]['dates'] = list(operator['SMAP'].sm_out.keys())
    results[i_real]['daily_diff'] = daily_diff
    results[i_real]['n_obs'] = n_obs
    
    save_dict_to_pickle(results,pickle_filename_tmp)
    
save_dict_to_pickle(results,pickle_filename)

# Finally plotting:
for i_real in results.keys():
    dates = np.array(results[i_real]['dates'])
    diff_max = 0.4

    str_date1 = str(results[i_real]['dates'][0])[0:10]
    str_date2 = str(results[i_real]['dates'][-1])[0:10]
    plt.figure()
    plt.pcolormesh(operator['SMAP'].grid_TSMP['lon_corner'],operator['SMAP'].grid_TSMP['lat_corner'],operator['SMAP'].grid_TSMP['lsm']==2,cmap=plt.cm.Greys,vmax=2 )

    plt.pcolormesh(operator['SMAP'].grid_TSMP.lon_corner,operator['SMAP'].grid_TSMP.lat_corner,
                   np.nanmean(results[i_real]['daily_diff'],axis=0),cmap=plt.cm.bwr,vmin=-diff_max,vmax=diff_max)
    cbar = plt.colorbar(extend='both')
    cbar.set_label('SMAP - TSMP [mm3/mm3]')
    plt.title('Mean mismatch between %s - %s' % (str_date1,str_date2 ) )
    plt.savefig(os.path.join(dir_figs,'mean_mismatch_%s_%s_%3.3i.pdf'%(str_date1,str_date2,i_real) ) )
    plt.savefig(os.path.join(dir_figs,'mean_mismatch_%s_%s_%3.3i.png'%(str_date1,str_date2,i_real) ) )


    plt.figure()
    plt.pcolormesh(operator['SMAP'].grid_TSMP.lon_corner,operator['SMAP'].grid_TSMP.lat_corner,
                   results[i_real]['n_obs'].sum(axis=0),cmap=plt.cm.viridis)
    cbar = plt.colorbar()
    cbar.set_label('Number of observations')
    plt.title('Number of SMAP observations between %s - %s' % (str_date1,str_date2 ) )
    plt.savefig(os.path.join(dir_figs,'nobs_%s_%s_%3.3i.pdf'%(str_date1,str_date2,i_real) ) )
    plt.savefig(os.path.join(dir_figs,'nobs_%s_%s_%3.3i.png'%(str_date1,str_date2,i_real) ) )

    i_train_start = 0
    i_train_end = np.where(dates == date_results_iter[1][-1])[0][0] + 1
    
    i_val_start = i_train_end 
    i_val_end = -1
    
    
    str_date1 = str(results[i_real]['dates'][i_train_start])[0:10]
    str_date2 = str(results[i_real]['dates'][i_train_end-1])[0:10]    
    plt.figure()
    plt.pcolormesh(operator['SMAP'].grid_TSMP['lon_corner'],operator['SMAP'].grid_TSMP['lat_corner'],operator['SMAP'].grid_TSMP['lsm']==2,cmap=plt.cm.Greys,vmax=2 )

    plt.pcolormesh(operator['SMAP'].grid_TSMP.lon_corner,operator['SMAP'].grid_TSMP.lat_corner,
                   np.nanmean(results[i_real]['daily_diff'][i_train_start:i_train_end,:,:],axis=0),cmap=plt.cm.bwr,vmin=-diff_max,vmax=diff_max)
    cbar = plt.colorbar(extend='both')
    cbar.set_label('SMAP - TSMP [mm3/mm3]')
    plt.title('Mean mismatch between %s - %s' % (str_date1,str_date2 ) )
    plt.savefig(os.path.join(dir_figs,'mean_mismatch_%s_%s_%3.3i.pdf'%(str_date1,str_date2,i_real) ) )
    plt.savefig(os.path.join(dir_figs,'mean_mismatch_%s_%s_%3.3i.png'%(str_date1,str_date2,i_real) ) )


    plt.figure()
    plt.pcolormesh(operator['SMAP'].grid_TSMP.lon_corner,operator['SMAP'].grid_TSMP.lat_corner,
                   results[i_real]['n_obs'][i_train_start:i_train_end,:,:].sum(axis=0),cmap=plt.cm.viridis)
    cbar = plt.colorbar()
    cbar.set_label('Number of observations')
    plt.title('Number of SMAP observations between %s - %s' % (str_date1,str_date2 ) )
    plt.savefig(os.path.join(dir_figs,'nobs_%s_%s_%3.3i.pdf'%(str_date1,str_date2,i_real) ) )
    plt.savefig(os.path.join(dir_figs,'nobs_%s_%s_%3.3i.png'%(str_date1,str_date2,i_real) ) )
    
    
    
    str_date1 = str(results[i_real]['dates'][i_val_start])[0:10]
    str_date2 = str(results[i_real]['dates'][i_val_end])[0:10]    
    plt.figure()
    plt.pcolormesh(operator['SMAP'].grid_TSMP['lon_corner'],operator['SMAP'].grid_TSMP['lat_corner'],operator['SMAP'].grid_TSMP['lsm']==2,cmap=plt.cm.Greys,vmax=2 )

    plt.pcolormesh(operator['SMAP'].grid_TSMP.lon_corner,operator['SMAP'].grid_TSMP.lat_corner,
                   np.nanmean(results[i_real]['daily_diff'][i_val_start:i_val_end,:,:],axis=0),cmap=plt.cm.bwr,vmin=-diff_max,vmax=diff_max)
    cbar = plt.colorbar(extend='both')
    cbar.set_label('SMAP - TSMP [mm3/mm3]')
    plt.title('Mean mismatch between %s - %s' % (str_date1,str_date2 ) )
    plt.savefig(os.path.join(dir_figs,'mean_mismatch_%s_%s_%3.3i.pdf'%(str_date1,str_date2,i_real) ) )
    plt.savefig(os.path.join(dir_figs,'mean_mismatch_%s_%s_%3.3i.png'%(str_date1,str_date2,i_real) ) )


    plt.figure()
    plt.pcolormesh(operator['SMAP'].grid_TSMP.lon_corner,operator['SMAP'].grid_TSMP.lat_corner,
                   results[i_real]['n_obs'][i_val_start:i_val_end,:,:].sum(axis=0),cmap=plt.cm.viridis)
    cbar = plt.colorbar()
    cbar.set_label('Number of observations')
    plt.title('Number of SMAP observations between %s - %s' % (str_date1,str_date2 ) )
    plt.savefig(os.path.join(dir_figs,'nobs_%s_%s_%3.3i.pdf'%(str_date1,str_date2,i_real) ) )
    plt.savefig(os.path.join(dir_figs,'nobs_%s_%s_%3.3i.png'%(str_date1,str_date2,i_real) ) )    