import numpy as np
import os
import re
import shutil
from datetime import datetime, timedelta
import pandas as pd
from glob import glob
import xarray as xr
import netCDF4

# from setup_parameters import setup_Ks,setup_Ks_tensor,setup_Ks_anom
# from generate_parameters import generate_Ks,generate_Ks_tensor,generate_Ks_anom
from run_realization_v2 import setup_submit_wait
from DA_operators import operator_clm_SMAP, operator_clm_FLX

from settings import settings_run,settings_clm,settings_pfl,settings_sbatch,settings_DA,settings_gen,date_results_binned,freq_output,date_range_noleap

# from multiprocessing import Pool
import multiprocessing as mp

from itertools import repeat
from scipy import sparse

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import time
import copy

from helpers import haversine_distance

os.environ['MKL_NUM_THREADS'] = '4'
os.nice(5)
'''
v2: test adjusting Ks tensor value
'''

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
                          dzeta_global=1.,dir_settings='.',factor_inflate_prior=1.,loc_type='distance',POL_eps=.5):
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
    print('Using %s localisation' % loc_type)
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
    
    # used for POL localisation:
    c_ii = np.var( (mat_M - mat_M.mean(axis=1)[:,np.newaxis]), axis=1, ddof=1) #(n_param,) -> estimate of C_M
    c_jj = np.var( (mat_D - mat_D.mean(axis=1)[:,np.newaxis]), axis=1, ddof=1) #(n_data,) -> estimate of C_D

    param_a = np.zeros(mat_M.shape)
    for i in range(n_param):
        
        # first localisation option: distance based, by using haversine distance & GC function
        if loc_type == 'distance':    
            if np.isnan(param_r_loc[i]):
                rho_i = dzeta_global*np.ones(n_data_) #no localisation
            else:
                r_loc = param_r_loc[i]
                # localisation using the Gaspari-Cohn localisation function:
                rho_i = GC(haversine_distance(param_latlon[i,:],data_latlon),r_loc)
        # second option: pseudo optimal localization
        # see e.g. Lacerda et al. (2019), Furrer et al. (2007)
        elif loc_type == 'POL':
            ci = del_M[i,:]@(del_D[:,:].T)
            rho_i = ci**2 / (ci**2 + (ci**2+(c_ii[i]*c_jj)/n_ensemble) )
            mask_zero = np.abs(ci) < POL_eps*np.sqrt(c_ii[i]*c_jj)
            rho_i[mask_zero] = 0
            
        else:
            print('Warning!! Localisation method unknown. Set to POL or distance')
            rho_i = np.ones(n_data_)

            
        K_i = del_M[i,:]@mat_X3
        K_rho_i = K_i * rho_i
        mat_X4 = K_rho_i @ (mat_Dobs - mat_D)
        param_a[i,:] = mat_M[i,:] + mat_X4
 
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



if __name__ == '__main__':


    plot_members_SMAP = 0 #set to int (iteration for which to plot members), or to False/True
    plot_members_FLX = True
    
    ### Unpack some of the settings into variables
    # Functions that are run to initialize the parameters to be assimilated. 
    # E.g. for spatial parameter fields, initialize the static fields (x,y,z) locations and the prior/uncertainty estimates
    param_setup = settings_DA['param_setup'] 
    # Functions that are run to generate realizations of parameters/state variables
    param_gen   = settings_DA['param_gen']
    # Define parameter names; parameters values are stored in (%s.param.npy % param_name) files
    param_names = settings_DA['param_names']

    # unpack some general settings
    data_nselect = settings_DA['n_data_max']
    n_parallel = settings_DA['n_parallel']
    n_parallel_setup = settings_DA['n_parallel_setup']
    n_ensemble = settings_DA['n_ensemble']
    n_iter = settings_DA['n_iter']
    dir_setup = settings_run['dir_setup']
    dir_template = settings_run['dir_template']

    # Some DA settings
    alpha = settings_DA['alpha']
    factor_inflate = settings_DA['factor_inflate']
    factor_inflate_prior = settings_DA['factor_inflate_prior']
    ksi=settings_DA['cutoff_svd']

    # Unpack/initialize settings on DA parameters
    data_names = settings_DA['data_names']
    data_var = copy.deepcopy(settings_DA['data_var']) #data_var is changed in the code; make therefore deepcopy
    data_mask = {'SMAP':None, #initialize dict
                 'FLX':None}
    
    '''
     1) Copy the folder template to the setup location if the destination does not exist
    '''
    if not os.path.exists(dir_setup):
        print('Copying folder template from %s to %s' % (dir_template,dir_setup) )
        shutil.copytree(dir_template,dir_setup)
    else:
        print('Continuing simulation in %s' % dir_setup)
    # os.chdir(dir_setup)

    # copy settings file for later use
    dir_settings = os.path.join(settings_run['dir_setup'],'settings')
    if not os.path.exists(dir_settings):
        os.mkdir(dir_settings)
        shutil.copy('settings.py',dir_settings)
        
    dir_figs = os.path.join(dir_setup,'figures')
    settings_run['dir_figs'] = dir_figs
    if not os.path.exists(dir_figs):
        print('Creating folder to store DA information: %s' % (dir_figs) )
        os.mkdir(dir_figs)

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
    #%% ----------- iteration loop -----------    
    # this comes in the iteration loop, e.g. iterate every year n times in the optimization
    for i_iter in np.arange(n_iter):

        # Set initialization flag to True in the first loop (for parameter initialization)
        if i_iter == 0:
            init = True
        else:
            init = False
        str_iter = 'i%3.3i' % i_iter
        dir_iter = os.path.join(dir_date,str_iter)
        if os.path.exists(os.path.join(dir_DA,'%s.param.000.%3.3i.000.npy'%(param_names[0],i_iter+1)) ): #check if the next iteration parameter files already exist
            print('Iteration %i seems to have finished succesfully, continuing with the next iteration...' % i_iter)
        else:
            if not os.path.exists(dir_iter):
                print('Creating folder for iteration %i: %s' % (i_iter,dir_iter) )
                os.mkdir(dir_iter)

            #%% ----------- ensemble member loop (parallel) -----------    
            # ensemble member (realization) loop, done in parallel
            # member 0 is reserved for the most likely parameter values

            settings_run['dir_iter'] = dir_iter
            settings_gen['i_date'] = i_date
            settings_gen['i_iter'] = i_iter
            settings_gen['param_gen'] = param_gen
            settings_gen['param_names'] = param_names
            settings_gen['param_r_loc'] = settings_DA['param_r_loc']
            # settings_gen['param_length'] = param_length

            if n_parallel_setup == 1:
                for i_real in np.arange(0,n_ensemble+1):
                    realize_parameters(i_real,settings_gen,settings_run,init=init)
            else:
                with mp.get_context("spawn").Pool(processes=n_parallel_setup) as pool: 
                    pool.starmap(worker_realize_parameters, zip(np.arange(0,n_ensemble+1),repeat(settings_gen),repeat(settings_run),repeat(init)) )
                    pool.close()
                    pool.join()
                    
            print('n_ensemble: %i' % n_ensemble, flush=True)
            # Aggregrate all parameter values into param_f
            param_f,param_names_all,param_latlon,param_r_loc = read_parameters(n_ensemble,settings_gen,settings_run)
            n_param = len(param_f)
            print('Amount of parameters to assimilate: %i' % n_param, flush=True)

            # Parallel submission -> if using single site perhaps make a single job script, using 1 node instead of many
            with mp.Pool(processes=n_parallel) as pool:
                pool.starmap(setup_submit_wait, zip(np.arange(0,n_ensemble+1),repeat(settings_run),repeat(settings_clm),
                                                   repeat(settings_pfl),repeat(settings_sbatch),repeat(date_results_iter)) )
            
            n_ensemble, reread_required = check_for_success(dir_iter,dir_DA,dir_settings,date_results_iter,n_ensemble)
            if reread_required:
                param_f,_,_,_ = read_parameters(n_ensemble,settings_gen,settings_run)
 
            # Measurement operators: map state vector onto measurement space
            # i.e. get the TSMP values at the SMAP or FLUXNET times/locations
            operator = {}
            data_measured = {}
            data_latlon = {}

            # Operator for SMAP
            operator['SMAP'] = operator_clm_SMAP(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_SMAP'],ignore_rivers=False)
            data_measured['SMAP'],_,data_latlon['SMAP'] = operator['SMAP'].get_measurements(date_results_iter,date_DA_start=date_start_sim,return_latlon=True)

            # Operator for FLUXNET
            operator['FLX'] = operator_clm_FLX(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_FLX'],ignore_rivers=False)
            data_measured['FLX'],data_var['FLX'],data_latlon['FLX'] = operator['FLX'].get_measurements(date_results_iter,date_DA_start=date_start_sim,return_latlon=True,variance=settings_DA['data_var']['FLX'])

            # mask observations to required amount given in data_nselect
            data_mask,data_indices,n_data,data_measured_masked,data_var_masked,data_latlon_masked = mask_observations(data_names,data_measured,data_var,data_latlon,data_nselect,data_mask,factor_inflate=factor_inflate)

            # get most likely parameter output, to track if the iterations are improving
            t0 = time.time()
            _ = operator['SMAP'].interpolate_model_results(0,settings_run,indices_z=[0,1],var='SOILLIQ')#[data_mask['SMAP']]
            operator['SMAP'].plot_results(i_iter,0,settings_run,indices_z=0,var='SOILLIQ',n_plots=12,dir_figs=settings_run['dir_figs'])
            _ = operator['FLX'].interpolate_model_results(0,settings_run,retain_history=True)#[data_mask['FLX']]
            operator['FLX'].plot_results(i_iter,0,settings_run,dir_figs=os.path.join(settings_run['dir_figs'],'FLX'))
            t1 = time.time()
            print('%f seconds for interpolating/plotting one ensemble member'%(t1-t0), flush=True)
            
            # 2) get the corresponding ensemble measurements ("forecast")
            print('Interpolating/plotting all ensemble members...', flush=True)
            t0 = time.time()
            data_f = np.zeros([n_data,n_ensemble])
            for i_real in np.arange(1,n_ensemble+1):   
                print('%i/%i' % (i_real,n_ensemble), flush=True)
                for var_ in data_names:
                    data_f[data_indices[var_][0]:data_indices[var_][1],i_real-1] = operator[var_].interpolate_model_results(i_real,settings_run)[data_mask[var_]]
                if 'SMAP' in data_names and (plot_members_SMAP==True or plot_members_SMAP==i_iter):
                    operator['SMAP'].plot_results(i_iter,i_real,settings_run,indices_z=0,var='SOILLIQ',n_plots=4)
            t1 = time.time()
            print('%f seconds for interpolating/plotting all ensemble members'%(t1-t0), flush=True)
            print('Shape of data array (data_f):')
            print(data_f.shape)
            
            if 'FLX' in data_names and plot_members_FLX:
                operator['FLX'].plot_all_results(i_iter,settings_run,dir_figs=os.path.join(settings_run['dir_figs'],'FLX'))
                

            print('Performing Kalman update...', flush=True)
            t0 = time.time()
            param_a,mean_mismatch_new,alpha = update_step_ESMDA_loc(param_f,data_f,data_measured_masked,data_var_masked,alpha,i_iter,n_iter,
                                                                    param_latlon=param_latlon,param_r_loc=param_r_loc,data_latlon=data_latlon_masked,ksi=ksi,
                                                                    dir_settings=dir_settings,dzeta_global=settings_DA['dzeta_global'],
                                                                    factor_inflate_prior=factor_inflate_prior,
                                                                    loc_type=settings_DA['loc_type'],POL_eps=settings_DA['POL_eps'])
            t1 = time.time()
            print('%f seconds for Kalman update'%(t1-t0), flush=True)

            write_parameters(param_a,settings_gen,settings_run)

            mismatch_iter.append(mean_mismatch_new)
            print('Mismatch (old forecast vs. new forecast): %3.3f -> %3.3f' % (mismatch_iter[-2],mismatch_iter[-1]), flush=True)

            plot_prior_post(param_f,param_a,param_names_all,i_iter,dir_figs=os.path.join(settings_run['dir_figs'],'params'))


    # last iteration: 
    # only compute result of most likely parameters if required
    # extend runs by an extra time period for validation
    # iter_n, R000: Most likely parameter run
    # iter_n, R001-R00N: Different ensemble members (N=n_ensemble)
    # iter_n, R00N+1: OL run, extended to include validation results
    
    if settings_DA['last_iter_ML_only']:
        init_list = [False,True] #DA run is already initialized, treat OL as non-initialized (i.e. read parameters from prior)
        run_prior = [False,True] #run OL with prior parameter values
        list_i_real = [0, n_ensemble+1]
    else: #evaluate the entire ensemble, plus the original prior settings (OL run)
        init_list = [False for i in range(n_ensemble+1)] #0...N: initialized, N+1 (OL) needs to be initialized from prior
        init_list.append(True)
        run_prior = [False for i in range(n_ensemble+1)]
        run_prior.append(True)
        list_i_real = [0+c for c in range(n_ensemble+2)] #[0...N,N+1]
        
    i_iter += 1
    str_iter = 'i%3.3i' % i_iter
    dir_iter = os.path.join(dir_date,str_iter)
    if not os.path.exists(dir_iter):
        print('Creating folder for iteration %i: %s' % (i_iter,dir_iter) )
        os.mkdir(dir_iter)

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

    for i_,i_real in enumerate(list_i_real):
        realize_parameters(i_real,settings_gen,settings_run,init=init_list[i_],run_prior=run_prior[i_])

    with mp.Pool(processes=n_parallel+1) as pool:
        pool.starmap(setup_submit_wait, zip(list_i_real,repeat(settings_run),repeat(settings_clm),
                                           repeat(settings_pfl),repeat(settings_sbatch),repeat(date_results_iter)) )

    operator = {}
    data_measured = {}
    operator['SMAP'] = operator_clm_SMAP(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_SMAP'],ignore_rivers=False)
    _ = operator['SMAP'].get_measurements(date_results_iter,date_DA_start=date_start_sim)
    operator['FLX'] = operator_clm_FLX(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_FLX'],ignore_rivers=False)
    _ = operator['FLX'].get_measurements(date_results_iter,date_DA_start=date_start_sim)

    dir_figures_validation = os.path.join(settings_run['dir_figs'],'validation')
    for i_real in list_i_real:

        if i_real == 0 or i_real == n_ensemble+1: #plot DA and OL
            _ = operator['SMAP'].interpolate_model_results(i_real,settings_run,indices_z=[0,1],var='SOILLIQ')
            _ = operator['FLX'].interpolate_model_results(i_real,settings_run,retain_history=True)
        
            operator['SMAP'].plot_results(i_iter,i_real,settings_run,dir_figs=dir_figures_validation,indices_z=0,var='SOILLIQ',n_plots=24*4)
            operator['FLX'].plot_results(i_iter,i_real,settings_run,dir_figs=dir_figures_validation)
        else:
            _ = operator['SMAP'].interpolate_model_results(i_real,settings_run,indices_z=[0,1],var='SOILLIQ')
            _ = operator['FLX'].interpolate_model_results(i_real,settings_run)
        
            
    if 'FLX' in data_names and plot_members_FLX:
        operator['FLX'].plot_all_results(i_iter,settings_run,dir_figs=dir_figures_validation)

    print('---------------------------------------')
    print('Wow, the DA actually finished, congrats')
    print('---------------------------------------')
    
    #         file_clm_last = sorted(glob(os.path.join(settings_run['dir_iter'],'R000/**/clm.clm2.r*')))[-1]
    #         file_pfl_last = sorted(glob(os.path.join(settings_run['dir_iter'],'R000/**/*.out.*.nc')))[-1]

    #         settings_pfl.update({'IC_file':file_pfl_last})
    #         settings_clm.update({'IC_file':file_clm_last})

    #         print('Resuming next iteration from CLM file %s' % file_clm_last)
    #         print('Resuming next iteration from PFL file %s' % file_pfl_last)

