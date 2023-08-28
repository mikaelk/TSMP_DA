import numpy as np
import os
import shutil
from datetime import datetime, timedelta
import pandas as pd
from glob import glob
import xarray as xr
import netCDF4

# from setup_parameters import setup_Ks,setup_Ks_tensor,setup_Ks_anom
# from generate_parameters import generate_Ks,generate_Ks_tensor,generate_Ks_anom
from run_realization_v2 import setup_submit_wait
from DA_operators import operator_clm_SMAP

from settings import settings_run,settings_clm,settings_pfl,settings_sbatch,settings_DA,settings_gen,date_results_binned

from multiprocessing import Pool
from itertools import repeat
from scipy import sparse

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

'''
v2: test adjusting Ks tensor value
'''

def realize_parameters(i_real,settings_gen,settings_run,init=True):
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    local_state = np.random.RandomState() #required for parallel processes in python
    dir_DA = settings_run['dir_DA']

    if not os.path.exists(dir_real):
        print('Creating parameter realizations for ensemble member %i' % i_real)
        print('Creating folder for realization %i: %s' % (i_real,dir_real), flush=True )
        os.mkdir(dir_real)

        if init:
            print('Initializing parameters from prior parameter settings')

            # Read parameter values + std, generate parameter realizations (i_real)
            for p_name, p_fn_gen in zip(settings_gen['param_names'],settings_gen['param_gen']):
                p_values = np.load(os.path.join(dir_DA,'%s.param.%3.3i.%3.3i.prior.npy'% (p_name,settings_gen['i_date'],settings_gen['i_iter']) ))
                p_mean = p_values[:,0]
                p_sigma = p_values[:,1]
                
                # ensemble member 0: most likely parameter values are used
                if i_real == 0:
                    p_real = p_mean.copy()
                else:
                    p_real = local_state.normal(p_mean,p_sigma)
                np.save(os.path.join(dir_DA,'%s.param.%3.3i.%3.3i.%3.3i'%(p_name,settings_gen['i_date'],settings_gen['i_iter'],i_real)),p_real)
                p_fn_gen(i_real,settings_gen,settings_run)
        else:
            print('Updating parameters from DA analysis')
            for p_name, p_fn_gen in zip(settings_gen['param_names'],settings_gen['param_gen']):
                p_fn_gen(i_real,settings_gen,settings_run)
        
        
def read_parameters(n_ensemble,settings_gen,settings_run):
    # read parameter values of the different ensemble members into an array
    for i1 in np.arange(n_ensemble):
        param_tmp = np.array([])
        
        if i1 == 0:
            for i2,p_name in enumerate(settings_gen['param_names']):
                param_ = np.load(os.path.join(settings_run['dir_DA'],'%s.param.%3.3i.%3.3i.%3.3i.npy'% (p_name,settings_gen['i_date'],settings_gen['i_iter'],i1+1) ))
                # settings_gen['param_length'][p_name] = len(param_)
                param_tmp = np.append(param_tmp,param_)
            param_all = param_tmp.copy()
    
        else:
            for i2,p_name in enumerate(settings_gen['param_names']):
                param_ = np.load(os.path.join(settings_run['dir_DA'],'%s.param.%3.3i.%3.3i.%3.3i.npy'% (p_name,settings_gen['i_date'],settings_gen['i_iter'],i1+1) ))
                param_tmp = np.append(param_tmp,param_)        
            param_all = np.vstack((param_all,param_tmp))
    return param_all.T 


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
        
        
def plot_results_ml(operator,settings_gen,settings_run):

    for date_ in operator.data_TSMP_ml.keys():

        str_date = str(date_.date())
        
        min_ = min(min(operator.data_TSMP_ml[date_]),min(operator.sm_out[date_]))
        max_ = max(max(operator.data_TSMP_ml[date_]),max(operator.sm_out[date_]))
        R = pearsonr(operator.data_TSMP_ml[date_],operator.sm_out[date_])[0]
        plt.figure(figsize=(5,5))
        plt.plot(operator.data_TSMP_ml[date_],operator.sm_out[date_],'o',alpha=.7,markersize=3)
        plt.plot([min_,max_],[min_,max_],'k:')
        plt.xlabel('Modelled soil moisture')
        plt.ylabel('SMAP soil moisture')
        plt.title('%s, R=%3.3f (mean param. values)' % (date_,R) )
        plt.savefig(os.path.join(settings_run['dir_figs'],'corr_%s_%3.3i.png'%(str_date,settings_gen['i_iter']) ) )

        plt.figure()
        plt.pcolormesh(operator.grid_TSMP['lon_corner'],operator.grid_TSMP['lat_corner'],operator.grid_TSMP['lsm']==2,cmap=plt.cm.Greys,vmax=2 )
        diff = operator.sm_out[date_] - operator.data_TSMP_ml[date_]
        # diff_max = max(np.abs(diff))
        diff_max = 0.4
        plt.scatter(operator.lons_out[date_],operator.lats_out[date_],s=10,c=diff,vmin=-diff_max,vmax=diff_max,cmap=plt.cm.bwr)
        cbar = plt.colorbar(extend='both')
        cbar.set_label('SMAP - TSMP (mean param. values)')
        plt.savefig(os.path.join(settings_run['dir_figs'],'mismatch_%s_%3.3i.png'%(str_date,settings_gen['i_iter']) ) )
  
        plt.close('all')
   
if __name__ == '__main__':

    data_names = ['SMAP']
    data_var = [0.15**2]

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


    if n_iter > 1:
        alpha = n_iter*np.ones(n_iter)
    elif n_iter == 8:
        alpha = np.array([20.719,19.0,17.,16.,15.,9.,5.,2.5])    
    else:
        alpha = [1.]

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

        # setup parameters: prior/uncertainties, + static properties based on the settings if necessary
        for fn in param_setup:
            fn(settings_gen,settings_run)

    # Read parameter length and put in dictionary here
    for param_ in param_names:
        settings_gen['param_length'][param_] = np.load(os.path.join(dir_DA,'%s.param.000.000.prior.npy' % param_) ).shape[0]

    #%% ----------- DA loop -----------


    #%% ----------- date loop -----------    
    # this comes in the date loop, e.g. perform the smoother over a period over 1 year:
    i_date = 0
    date_results_iter = date_results_binned[i_date]
    date_start_sim = date_results_binned[i_date][0][0]#datetime(2019,1,2,12,0,0)
    date_end_sim = date_results_binned[i_date][-1][-1]#datetime(2019,12,31,12,0,0)
    str_date = str(date_start_sim.date()).replace('-','') + '-' + str(date_end_sim.date()).replace('-','')
    dir_date = os.path.join(dir_setup,str_date)
    if not os.path.exists(dir_date):
        print('Creating folder for dates %s: %s' % (str_date,dir_date) )
        os.mkdir(dir_date)

    ## TEMP
    date_DA_start = date_end_sim - timedelta(days=30) # spinup, only assimilate last 30 days

    mismatch_iter = [0]
    #%% ----------- iteration loop -----------    
    # this comes in the iteration loop, e.g. interate every year n times in the optimization
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
            # settings_gen['param_length'] = param_length

            with Pool(processes=n_parallel_setup) as pool:
                pool.starmap(realize_parameters, zip(np.arange(0,n_ensemble+1),repeat(settings_gen),repeat(settings_run),repeat(init)) )


            # Aggregrate all parameter values into param_f
            param_f = read_parameters(n_ensemble,settings_gen,settings_run)
            n_param = len(param_f)

            with Pool(processes=n_parallel) as pool:
                pool.starmap(setup_submit_wait, zip(np.arange(0,n_ensemble+1),repeat(settings_run),repeat(settings_clm),
                                                   repeat(settings_pfl),repeat(settings_sbatch),repeat(date_results_iter)) )


            # Measurement operator: map state vector onto measurement space
            # i.e. get the TSMP values at the SMAP times/locations
            operator = operator_clm_SMAP(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_SMAP'])
            # 1) get the observed quantities, and corresponding lon/lat/time    
            data_measured = operator.get_measurements(date_results_iter,date_DA_start=date_DA_start)

            ### TEMPORARY: reduce number of measurements
            n_select = 20000
            frac_select = min((n_select / len(data_measured)),.99)
            mask_measured = np.random.choice([0,1],size=len(data_measured),p=[1-frac_select,frac_select]).astype(bool) 
            data_measured=data_measured[mask_measured]

            n_data = len(data_measured)
            C_D = data_var[0]*sparse.eye(n_data)    

            # 2) get the corresponding ensemble measurements ("forecast")
            data_f = np.zeros([n_data,n_ensemble])
            for i_real in np.arange(1,n_ensemble+1):    
                data_f[:,i_real-1] = operator.interpolate_model_results(i_real,settings_run)[mask_measured]
            # get most likely parameter output as well, to track if the iterations are improving
            data_ml = operator.interpolate_model_results(0,settings_run)[mask_measured]
            plot_results_ml(operator,settings_gen,settings_run)


            # 3) construct covariance matrices based on ensemble of parameters and results (data)
            C_MD = np.zeros([n_param,n_data],dtype=np.float32)
            C_DD = np.zeros([n_data,n_data],dtype=np.float32)
            param_mean = param_f.mean(axis=1)
            data_mean = data_f.mean(axis=1)        
            param_delta = np.zeros([n_param,n_ensemble])
            data_delta = np.zeros([n_data,n_ensemble])
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

                z_d = np.random.normal(0,1,n_data)
                data_perturbed = data_measured + np.sqrt(alpha[i_iter])*np.sqrt(C_D.diagonal())*z_d

                mismatch = data_perturbed - data_f[:,i_real]

                mean_mismatch_new += np.sum(mismatch**2)

                # forecast -> analysis
                param_a[:,i_real] = param_f[:,i_real] + np.dot(KG,mismatch)

            write_parameters(param_a,settings_gen,settings_run)

            mean_mismatch_new /= n_ensemble
            mismatch_iter.append(mean_mismatch_new)
            print('Mismatch: %3.3f -> %3.3f' % (mismatch_iter[-2],mismatch_iter[-1]))

    #         file_clm_last = sorted(glob(os.path.join(settings_run['dir_iter'],'R000/**/clm.clm2.r*')))[-1]
    #         file_pfl_last = sorted(glob(os.path.join(settings_run['dir_iter'],'R000/**/*.out.*.nc')))[-1]

    #         settings_pfl.update({'IC_file':file_pfl_last})
    #         settings_clm.update({'IC_file':file_clm_last})

    #         print('Resuming next iteration from CLM file %s' % file_clm_last)
    #         print('Resuming next iteration from PFL file %s' % file_pfl_last)

