from datetime import datetime, timedelta
import pandas as pd

def bin_dates_by_restart_dates(date_results,date_restarts,spinup=False):
    '''
    Function that bins date_results (dates for which results are written) into bins defined by date_restarts
    Each bin end correspons to the next bin start (date_results_binned[0][-1] == date_results_binned[1][0])
    Warning: last bin can contain a different amount of result files, since the total time does not necessarily bin into n integer intervals
    '''
    date_results_binned = [[]] #bin the simulation dates into the periods defined by date_restarts
    c = 0
    date_new_bin = date_restarts[c+1]
    for date_ in date_results:
        if date_ >= date_new_bin and date_ < date_restarts[-1] and date_ < date_results[-1]:
            date_results_binned[c].append(date_) #add the first date of the next bin to the date array to have 'overlapping' dates for the restart
            c+=1
            date_new_bin = date_restarts[c+1]
            date_results_binned.append([])
        date_results_binned[c].append(date_)    
   
    if spinup:
        assert date_end == date_results[-1], 'Choose an output frequency that fits n (integer) times inside of the restart frequency, e.g. 1 year sim with restart "AS" and output "MS". %s' % date_results
        assert len(date_results_binned) == 1
        date_results_binned = [date_results_binned[0]]*spinup

    return date_results_binned


'''
### USER INPUT ###
'''     
date_start = datetime(2019,1,1,12,0,0)
date_end = datetime(2019,2,1,12,0,0)
freq_output = '3d' 
freq_iter = 'MS'
freq_restart = 'MS' #e.g. '7d','AS','MS' # AS = annual, start of year (see pandas date_range freq options)

time_couple = timedelta(seconds=900) # coupling, don't change this - or check coup_oas.tcl carefully (e.g. baseunit) - pfl units are in hours
nx = 111 #111,222,444
ny = 108 #108,216,432

settings_run={'dir_forcing':'/p/scratch/cjibg36/kaandorp2/data/ERA5_EUR-44_CLM', #folder containing CLM forcing files
            'dir_setup':'/p/scratch/cjibg36/kaandorp2/TSMP_results/TSMP_patched/DA_tsmp_cordex_%ix%i' % (nx,ny), #folder in which the case will be run
            'dir_build':'/p/project/cjibg36/kaandorp2/TSMP_patched/', #required for parflow files
            'dir_binaries':'/p/project/cjibg36/kaandorp2/TSMP_patched/bin/JUWELS_3.1.0MCT_clm-pfl', #folder from which parflow/clm binaries are to be copied
            'dir_store':None, #files are moved here after the run is finished
            'dir_template':'/p/project/cjibg36/kaandorp2/TSMP_setups/setup_DA_tsmp_cordex_%ix%i/' % (nx,ny), #folder containing all clm/pfl/oasis/namelist files, where everything with 2 underscores (__variable__) needs to be filled out 
            'spinup':False, # integer (how many times to repeat the interval from date_start to date_end) or set to False
            'init_restart':True} #Set to true if you have initial restart files available for CLM/PFL, and set them correctly in the 2 lines below

IC_file_CLM = '/p/scratch/cjibg36/kaandorp2/TSMP_results/TSMP_patched/tsmp_cordex_spinup_111x108/run_009_20090101-20100101/clm.clm2.r.2010-01-01-43200.nc'
IC_file_ParFlow = '/p/scratch/cjibg36/kaandorp2/TSMP_results/TSMP_patched/tsmp_cordex_spinup_111x108/run_009_20090101-20100101/cordex111x108_20090101-20100101.out.00011.nc'

#---Some options for n_nodes(n_cores)=1(48),2(96),3(144),4(192),6(288)
n_proc_pfl_x = 6 #6,9,11,12,15
n_proc_pfl_y = 6 #6,9,11,12,15
n_proc_pfl_z = 1
n_proc_clm = 12 #12,15,23,48,63
sbatch_account = 'jibg36'
sbatch_partition = 'devel' #batch
sbatch_time = '0-01:00:00' #1-00:00:00 
sbatch_check_sec = 60*5 #check every n seconds if the simulation is done

'''
### END USER INPUT ###
'''  

# EU-cordex specific: domain dimensions
x_tot = 5550000
y_tot = 5400000

date_iterations = pd.date_range(date_start,date_end,freq=freq_iter)
date_restarts = pd.date_range(date_start,date_end,freq=freq_restart)
date_results = pd.date_range(date_start,date_end,freq=freq_output)

# We put the dates into lists, with the following hierarchy:
# 1) the time span (date_start to date_end) is separated into iteration time windows: e.g. we can run a smoother that repeats 1 year simulations a bunch of times over a decade (date_iterations)
# 2) each iteration can be subdivided into several restarts: e.g. divide 1-year simulations into 1-month batches for high res runs (date_restarts)
# 3) output is written at a frequency of 'freq_output' (date_results)
date_iter_binned = bin_dates_by_restart_dates(date_results,date_iterations)
date_restarts_binned = bin_dates_by_restart_dates(date_restarts,date_iterations)
date_results_binned = [bin_dates_by_restart_dates(date_iter_binned[i1],date_restarts_binned[i1]) for i1 in range(len(date_iter_binned))] 

time_dump = date_results[1]-date_results[0] # Not a perfect solution, can lead to the output frequency being too low (with freq_output='MS' for example)

settings_clm = {'t_dump':time_dump,
                't_couple':time_couple,
                'dir_forcing':settings_run['dir_forcing'],
                'nx':nx,
                'ny':ny,
                'n_proc_clm':n_proc_clm,
               'IC_file':IC_file_CLM}

settings_pfl = {'t_dump':time_dump,
                't_couple':time_couple,
                'nx':nx,
                'ny':ny,
                'dx':('%.1f' % (x_tot/nx)),
                'dy':('%.1f' % (y_tot/ny)),
                'n_proc_pfl_x':n_proc_pfl_x,
                'n_proc_pfl_y':n_proc_pfl_y,
                'n_proc_pfl_z':n_proc_pfl_z,
               'IC_file':IC_file_ParFlow}
               
settings_sbatch = {'sbatch_account':sbatch_account,
                  'sbatch_partition':sbatch_partition,
                  'sbatch_time':sbatch_time,
                  'sbatch_check_sec':sbatch_check_sec}     