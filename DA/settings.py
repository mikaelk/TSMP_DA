from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from helpers import bin_dates_by_restart_dates, date_range_noleap

# Below, import the parameter setup/generation functions that you defined 
from setup_parameters import setup_sandfrac_anom, setup_clayfrac_anom, setup_orgfrac_anom, setup_medlyn_slope, setup_medlyn_intercept, setup_medlyn_slope_v2, setup_medlyn_intercept_v2, setup_fff, setup_orgmax, setup_orgmax_v2
from setup_parameters import setup_om_hydraulic, setup_h2o_canopy_max, setup_kmax, setup_kmax_v2, setup_mineral_hydraulic, setup_luna

from generate_parameters import generate_sandfrac_anom, generate_clayfrac_anom, generate_orgfrac_anom, generate_medlyn_slope, generate_medlyn_intercept, generate_medlyn_slope_v2, generate_medlyn_intercept_v2, generate_fff, generate_orgmax, generate_orgmax_v2
from generate_parameters import generate_om_hydraulic, generate_h2o_canopy_max, generate_kmax, generate_kmax_v2, generate_mineral_hydraulic, generate_luna

# some old parflow functions that are not used anymore, but can be useful in the future
from setup_parameters import setup_Ks,setup_Ks_tensor,setup_Ks_anom,setup_n_anom,setup_a_anom,setup_poros_anom,setup_slope_anom
from generate_parameters import generate_Ks,generate_Ks_tensor,generate_Ks_anom,generate_n_anom,generate_a_anom,generate_poros_anom,generate_slope_anom


'''
### USER INPUT ###
'''  
date_start = datetime(2019,5,1,20,0,0)
date_end = datetime(2019,6,1,20,0,0)
freq_output = '3d'#'3d' 
freq_iter = 1 # int or string, e.g. 'AS','3MS','AS-MAY'  Set this to 1, unless you want to run iterative DA
freq_restart = 1 # int or string, e.g. '7d','AS','MS' # AS = annual, start of year (see pandas date_range freq options)
ndays_spinup = 3*30 #3*30 # set to multiple of freq_output! or to None
ndays_validation = 30# 12*30 # after parameter calibration, run for n days to check validation data

time_couple = timedelta(seconds=3600) # coupling, don't change this - or check coup_oas.tcl carefully (e.g. baseunit) - pfl units are in hours
nx = 444 #111,222,444
ny = 432 #108,216,432
nz = 30 #30 for eCLM, 15 for CLM3.5

settings_run={'models': 'eCLM', #model components to include ('eCLM' or 'CLM3.5-PFL', rest to be done..)
              'mode': 'DA', #Open Loop (OL), or with DA (adjust settings_DA, settings_gen)
              'dir_forcing':'/p/scratch/cjibg36/kaandorp2/data/ERA5_EUR-11_CLM_v2', #folder containing CLM forcing files
              'dir_setup':'/p/scratch/cjibg36/kaandorp2/TSMP_results/eTSMP/DA_test_clean_v1', #folder in which the case will be run
              'dir_build':'/p/project/cjibg36/kaandorp2/eCLM_params/', #required for parflow files
              'dir_binaries':'/p/project/cjibg36/kaandorp2/eCLM_params/eclm/bin/', #folder from which parflow/clm binaries are to be copied
              'dir_template':'/p/project/cjibg36/kaandorp2/eTSMP_setups/setup_eclm_cordex_444x432_v8/', #folder containing all clm/pfl/oasis/namelist files, where everything with 2 underscores (__variable__) needs to be filled out 
              'spinup':False, # integer (how many times to repeat the interval from date_start to date_end) or set to False
              'init_restart':True, #Set to true if you have initial restart files available for CLM/PFL, and set them correctly in the 2 lines below
              'env_file':'/p/project/cjibg36/kaandorp2/eTSMP/env/jsc.2023_Intel.sh', # file containing modules, os.path.join(dir_build,'bldsva/machines/JUWELS/loadenvs.Intel'
              'files_remove':[],
              'ndays_spinup':ndays_spinup,
              'ndays_validation':ndays_validation,
              'remove_hist_files':['h1','h2']} #set to None, or one of the history filestream names if they are to be removed

# Initial condition files for CLM/PFL
IC_file_CLM = '/p/scratch/cjibg36/kaandorp2/TSMP_results/eTSMP/OL_eclm_cordex_444x432/R000/run_009_20180101-20190101/EU11.clm2.r.2019-01-01-00000.nc'
IC_file_ParFlow = False

# SLURM settings. At the moment there is 1 CLM simulation per node, parflow settings are ignored when running CLM only.
#I've commented out some possible options for n_nodes(n_cores)=1(48),2(96),3(144),4(192),6(288)
n_proc_pfl_x = 9 #6,9,11,12,15
n_proc_pfl_y = 9 #6,9,11,12,15
n_proc_pfl_z = 1
n_proc_clm = 48 #12,15,23,48,63
sbatch_account = 'jibg36'
sbatch_partition = 'devel' #batch or devel (for short <1h runs)
sbatch_time = '0-01:00:00' #1-00:00:00 
sbatch_check_sec = 60*5 #check every n seconds if the simulation is done


#---Options for the Data Assimilation
settings_DA={'param_setup':[setup_orgmax_v2, setup_fff,setup_h2o_canopy_max,
                            setup_mineral_hydraulic,setup_om_hydraulic,
                            setup_kmax_v2,setup_medlyn_slope_v2,setup_medlyn_intercept_v2,setup_luna,
                            setup_sandfrac_anom, setup_clayfrac_anom, setup_orgfrac_anom ],
             'param_gen':[generate_orgmax_v2, generate_fff,generate_h2o_canopy_max,
                          generate_mineral_hydraulic,generate_om_hydraulic,
                          generate_kmax_v2,generate_medlyn_slope_v2,generate_medlyn_intercept_v2,generate_luna,
                          generate_sandfrac_anom, generate_clayfrac_anom, generate_orgfrac_anom ],
             'param_names':['orgmax_v2','fff','h2o_canopy_max',
                            'mineral_hydraulic','om_hydraulic',
                            'kmax_v2','medlyn_slope_v2','medlyn_intercept_v2','luna',
                            'sandfrac_anom', 'clayfrac_anom', 'orgfrac_anom'],
             'param_r_loc':[np.nan, np.nan, np.nan,
                            np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan,
                            12.500*16,12.500*16,12.500*16], #localisation radius in km
             'n_parallel':4,  # set to n_ensemble+1 for full efficiency
             'n_parallel_setup':4, # setup is done on login node, limit the nr of processes
             'n_ensemble':3,
             'n_iter':2,
             'last_iter_ML_only':False, #evaluate most likely parameter set for last iteration only (not entire ensemble)
             'data_names':['SMAP','FLX'], #which datasets to assimilate
             'n_data_max':{'SMAP':1e6,'FLX':1e6}, #limit maximum data to assimilate per dataset
             'data_var':{'SMAP':0.04**2,'FLX':None}, #data variance: constant, monthly, or calculate in the operator (None)
             'alpha':None, # prescribe inflation factors (list with floats), or calculate on the fly (None)
             'factor_inflate':{'SMAP':1.0,'FLX':1.0}, # add additional inflation to measurements. I set these to 1 in the end, and use the option below instead
             'factor_inflate_prior':1.05, # inflate the ensemble spread (i.e. deviation from ensemble mean), see e.g. doi.org/10.1175/2008MWR2691.1
             'loc_type':'distance', #type of localisation applied; POL or distance (set param_r_loc). POL does not seem to work better at first sight
             'dzeta_global':.4, #tapering factor
             'POL_eps':.8, #epsilon when using POL localization
             'cutoff_svd':.9, # discard small singular values, smaller = more discarding
             'file_lsm':'/p/project/cjibg36/kaandorp2/TSMP_setups/static/EUR-11_TSMP_FZJ-IBG3_444x432_LAND-LAKE-SEA-MASK.nc',
             'file_corner':'/p/project/cjibg36/kaandorp2/TSMP_setups/static/EUR-11_444x432_corners_curvi_Tair.nc',
             'folder_SMAP':'/p/scratch/cjibg36/kaandorp2/data/SMAP/',
             'folder_FLX':'/p/scratch/cjibg36/kaandorp2/data/FLUXNET'} 

# settings required for the parameter generation in setup_parameters and generate_parameters
settings_gen = {'dir_clm_surf':os.path.join(settings_run['dir_setup'],'input_clm'),
                'file_clm_surf':'surfdata_EUR-11_hist_16pfts_Irrig_CMIP6_simyr2000_c230808_GLC2000.nc',
                'texture_sample_xy':16,
                'texture_start_x':8,
                'texture_start_y':8,
                'texture_sample_z':None,
                'texture_plot':True,
                'texture_anom_l':16,
                'texture_anom_l_bounds':'fixed', #(1.2,120)
                'texture_nu':0.5, #0.5 for exponential kernel
                'param_length':{},
                'perturb_frac_std':0.1,
                'file_indi':os.path.join(settings_run['dir_template'],'input_pf/EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_%ix%i_INDICATOR_regridded_rescaled_SoilGrids250-v2017_BGR3_alv.sa'%(nx,ny)),
                'Ks_sample_xy': 10, #OLD PFL settings below
                'Ks_sample_z': 5,
                'Ks_mode':'ml',
                'Ks_plot':True,
                'a_sample_xy': 10,
                'a_sample_z': 5,
                'a_plot':True,
                'n_sample_xy': 10,
                'n_sample_z': 5,
                'n_plot':True,
                'poros_sample_xy': 10,
                'poros_sample_z': 5,
                'poros_plot':True,
                'file_slopex':os.path.join(settings_run['dir_template'],'input_pf/slopex.sa'),
                'file_slopey':os.path.join(settings_run['dir_template'],'input_pf/slopey.sa'),
                'slope_sample_xy': 10,
                'slope_plot':True} #for parameters without a given range, perturb e.g. 10% (0.1) or 20% (0.2)

'''
### END USER INPUT ###
'''  
# make sure the parameter settings have equal length
assert(len(settings_DA['param_setup'])==len(settings_DA['param_gen'])==len(settings_DA['param_names'])==len(settings_DA['param_r_loc']))

# EU-cordex specific: domain dimensions
# x_tot = 5550000
# y_tot = 5400000
# dx = x_tot/nx
# dy = y_tot/ny
dx = 12500.
dy = 12500.
dz = 1.

if type(freq_iter) == str:
    date_iterations = date_range_noleap(date_start,date_end,freq=freq_iter)
elif type(freq_iter) == int:
    date_iterations = date_range_noleap(date_start,date_end,periods=freq_iter+1)
    
if type(freq_restart) == str:
    date_restarts = date_range_noleap(date_start,date_end,freq=freq_restart)
elif type(freq_restart) == int:
    date_restarts = date_range_noleap(date_start,date_end,periods=freq_restart+1)
    
date_results = date_range_noleap(date_start,date_end,freq=freq_output)

# We put the dates into lists, with the following hierarchy:
# 1) the time span (date_start to date_end) is separated into iteration time windows: e.g. we can run a smoother that repeats 1 year simulations a bunch of times over a decade (date_iterations)
# 2) each iteration can be subdivided into several restarts: e.g. divide 1-year simulations into 1-month batches for high res runs (date_restarts)
# 3) output is written at a frequency of 'freq_output' (date_results)
date_iter_binned = bin_dates_by_restart_dates(date_results,date_iterations,spinup=settings_run['spinup'])
date_restarts_binned = bin_dates_by_restart_dates(date_restarts,date_iterations,spinup=settings_run['spinup'],avoid_big_bins=False)
date_results_binned = [bin_dates_by_restart_dates(date_iter_binned[i1],date_restarts_binned[i1],avoid_big_bins=True) for i1 in range(len(date_iter_binned))] 

time_dump = date_results[1]-date_results[0] # Not a perfect solution, can lead to the output frequency being too low (with freq_output='MS' for example)

settings_clm = {'t_dump':time_dump,
                't_dump2':timedelta(days=1), #variables that need to be written at a specific frequency (e.g. QFLX_EVAP_TOT for comparison to daily fluxnet data)
                'vars_dump':['SOILLIQ:I'],
                'vars_dump2':['QFLX_EVAP_TOT'], #variables that need to be written daily (e.g. FLX)
                'vars_dump_val':['SOILLIQ:I','TWS','H2OSOI','SOILLIQ','SOILICE','TSKIN','TSOI','Qh','Qle'], #for validation runs output extra
                't_couple':time_couple,
                'dir_forcing':settings_run['dir_forcing'],
                'nx':nx,
                'ny':ny,
                'n_proc_clm':n_proc_clm,
                'IC_file':IC_file_CLM,
                'init_interp':False, #eCLM option: interpolate the restart file. Must be true when going from eCLM only to eCLM/ParFlow runs, or runs with different masks
                'dir_common_eclm':'/p/project/cjibg36/kaandorp2/eTSMP_setups/common_eclm_files',
                'param_names':settings_DA['param_names']} #eCLM option: avoid copying many large files

settings_pfl = {'t_dump':time_dump,
                't_couple':time_couple,
                'nx':nx,
                'ny':ny,
                'nz':nz,
                'dx':('%.1f' % dx),
                'dy':('%.1f' % dy),
                'dz':('%.1f' % dz),
                'n_proc_pfl_x':n_proc_pfl_x,
                'n_proc_pfl_y':n_proc_pfl_y,
                'n_proc_pfl_z':n_proc_pfl_z,
                'IC_file':IC_file_ParFlow,
                'param_names':settings_DA['param_names']}

settings_sbatch = {'sbatch_account':sbatch_account,
                  'sbatch_partition':sbatch_partition,
                  'sbatch_time':sbatch_time,
                  'sbatch_check_sec':sbatch_check_sec}     
