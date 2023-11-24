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

from multiprocessing import Pool
from itertools import repeat
from scipy import sparse

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import time

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
        print('...')
        time.sleep(3)
        
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
        
        if len(operator.data_TSMP_ml[date_])>0:

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

            
def check_for_success(dir_iter,dir_DA,dir_settings,date_results_iter,n_ensemble):

    def change_setting(filename, key, new_value):
        # Escape special characters in the key
        escaped_key = re.escape(key)

        # Define the pattern to match
        pattern = re.compile(r"('{}'\s*:\s*)(\d+)".format(escaped_key))

        # Read the content of the file
        with open(filename, 'r') as file:
            content = file.read()

        # Use the pattern to find and replace the matched value
        content = pattern.sub(r"\g<1>{}".format(new_value), content)

        # Write the updated content back to the file
        with open(filename, 'w') as file:
            file.write(content)
        
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

            paramfiles_source = glob(os.path.join(dir_DA,'*.%3.3i.npy'%i_source))
            paramfiles_source_tmp = [file_ + '_old' for file_ in paramfiles_source]
            paramfiles_dest = glob(os.path.join(dir_DA,'*.%3.3i.npy'%i_dest))
            paramfiles_dest_tmp = [file_ + '_old' for file_ in paramfiles_dest]

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


def update_step_ESMDA(param_f,data_f,data_measured,data_var,alpha,i_iter):
    print('Calculating KG and performing parameter update...')
    assert data_f.shape[0] == len(data_measured)
    n_data = data_f.shape[0]
    n_ensemble = data_f.shape[1]
    n_param = param_f.shape[0]
    
    # 3) construct covariance matrices based on ensemble of parameters and results (data)
    C_D = data_var['SMAP']*sparse.eye(n_data)    
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

    mean_mismatch_new /= n_ensemble
    
    return param_a, mean_mismatch_new


if __name__ == '__main__':

    # which variables to assimilate. If uncertainties are assumed constant, prescribe
    data_names = ['SMAP','FLX']
    data_var = {'SMAP':0.15**2,
                'FLX':None}

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

            # with Pool(processes=n_parallel_setup) as pool:
            #     pool.starmap(realize_parameters, zip(np.arange(0,n_ensemble+1),repeat(settings_gen),repeat(settings_run),repeat(init)) )
            for i_real in np.arange(0,n_ensemble+1):
                realize_parameters(i_real,settings_gen,settings_run,init=init)

            print('n_ensemble: %i' % n_ensemble)
            # Aggregrate all parameter values into param_f
            param_f = read_parameters(n_ensemble,settings_gen,settings_run)
            n_param = len(param_f)
            print('Amount of parameters to assimilate: %i' % n_param)

            with Pool(processes=n_parallel) as pool:
                pool.starmap(setup_submit_wait, zip(np.arange(0,n_ensemble+1),repeat(settings_run),repeat(settings_clm),
                                                   repeat(settings_pfl),repeat(settings_sbatch),repeat(date_results_iter)) )
            
            n_ensemble, reread_required = check_for_success(dir_iter,dir_DA,dir_settings,date_results_iter,n_ensemble)
            if reread_required:
                param_f = read_parameters(n_ensemble,settings_gen,settings_run)
 
            # Measurement operator: map state vector onto measurement space
            # i.e. get the TSMP values at the SMAP times/locations
            operator = operator_clm_SMAP(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_SMAP'],ignore_rivers=False)
            # 1) get the observed quantities, and corresponding lon/lat/time    
            data_measured = operator.get_measurements(date_results_iter,date_DA_start=date_start_sim)

            # Operators for validation data; FLUXNET
            operator_FLX = operator_clm_FLX(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_FLX'],ignore_rivers=False)
            data_measured_FLX = operator_FLX.get_measurements(date_results_iter,date_DA_start=date_start_sim)
            
            ### TEMPORARY: reduce number of measurements
            n_select = 30000
            if len(data_measured) < n_select:
                mask_measured = np.ones(len(data_measured),dtype=bool)
            else:
                frac_select = min((n_select / len(data_measured)),.99)
                mask_measured = np.random.choice([0,1],size=len(data_measured),p=[1-frac_select,frac_select]).astype(bool) 
            data_measured=data_measured[mask_measured]

            n_data = len(data_measured)

            # get most likely parameter output, to track if the iterations are improving
            data_ml = operator.interpolate_model_results(0,settings_run,indices_z=[0,1],var='SOILLIQ')[mask_measured]
            operator.plot_results(i_iter,0,settings_run,indices_z=0,var='SOILLIQ',n_plots=12,dir_figs=settings_run['dir_figs'])
            operator_FLX.interpolate_model_results(0,settings_run)
            operator_FLX.plot_results(i_iter,0,settings_run,dir_figs=os.path.join(settings_run['dir_figs'],'FLX'))
            
            # 2) get the corresponding ensemble measurements ("forecast")
            data_f = np.zeros([n_data,n_ensemble])
            for i_real in np.arange(1,n_ensemble+1):    
                data_f[:,i_real-1] = operator.interpolate_model_results(i_real,settings_run,indices_z=[0,1],var='SOILLIQ')[mask_measured]
                operator.plot_results(i_iter,i_real,settings_run,indices_z=0,var='SOILLIQ',n_plots=4)

            # plot_results_ml(operator,settings_gen,settings_run)


            param_a,mean_mismatch_new = update_step_ESMDA(param_f,data_f,data_measured,data_var,alpha,i_iter)
         
            write_parameters(param_a,settings_gen,settings_run)
   
            mismatch_iter.append(mean_mismatch_new)
            print('Mismatch: %3.3f -> %3.3f' % (mismatch_iter[-2],mismatch_iter[-1]))

        
        
    # last iteration: only compute result of most likely parameters,
    # extend runs by an extra time period for validation
    # iter_n, R000: DA run (most likely parameters)
    # iter_n, R001: OL run, extended to include validation results
    init_list = [False,True] #DA run is already initialized, treat OL as non-initialized
    run_prior = [False,True] #run OL with prior parameter values

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

    for i_real in np.arange(0,2):
        realize_parameters(i_real,settings_gen,settings_run,init=init_list[i_real],run_prior=run_prior[i_real])



    with Pool(processes=2) as pool:
        pool.starmap(setup_submit_wait, zip(np.arange(0,2),repeat(settings_run),repeat(settings_clm),
                                           repeat(settings_pfl),repeat(settings_sbatch),repeat(date_results_iter)) )


    operator = operator_clm_SMAP(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_SMAP'],ignore_rivers=False)
    data_measured = operator.get_measurements(date_results_iter,date_DA_start=date_start_sim)
    operator_FLX = operator_clm_FLX(settings_DA['file_lsm'],settings_DA['file_corner'],settings_DA['folder_FLX'],ignore_rivers=False)
    data_measured_FLX = operator_FLX.get_measurements(date_results_iter,date_DA_start=date_start_sim)

    for i_real in np.arange(0,2):
        _ = operator.interpolate_model_results(i_real,settings_run,indices_z=[0,1],var='SOILLIQ')
        operator.plot_results(i_iter,i_real,settings_run,indices_z=0,var='SOILLIQ',n_plots=24*4)
        operator_FLX.interpolate_model_results(i_real,settings_run)
        operator_FLX.plot_results(i_iter,i_real,settings_run,dir_figs=os.path.join(settings_run['dir_figs'],'FLX'))
            
      
    print('---------------------------------------')
    print('Wow, the DA actually finished, congrats')
    print('---------------------------------------')
    
    #         file_clm_last = sorted(glob(os.path.join(settings_run['dir_iter'],'R000/**/clm.clm2.r*')))[-1]
    #         file_pfl_last = sorted(glob(os.path.join(settings_run['dir_iter'],'R000/**/*.out.*.nc')))[-1]

    #         settings_pfl.update({'IC_file':file_pfl_last})
    #         settings_clm.update({'IC_file':file_clm_last})

    #         print('Resuming next iteration from CLM file %s' % file_clm_last)
    #         print('Resuming next iteration from PFL file %s' % file_pfl_last)

