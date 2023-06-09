from datetime import datetime, timedelta
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import xarray as xr
import os
import shutil
import time
from glob import glob

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
        if date_ >= date_new_bin and date_ < date_restarts[-1]:
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

def copy_binaries(dir_binaries,dir_run):
    print('Copy binaries from %s' % dir_binaries)

    shutil.copyfile(os.path.join(dir_binaries,'clm'), os.path.join(dir_run,'clm'))
    os.chmod(os.path.join(dir_run,'clm'),0o755) # read/execute permissions for all

    shutil.copyfile(os.path.join(dir_binaries,'parflow'), os.path.join(dir_run,'parflow'))
    os.chmod(os.path.join(dir_run,'parflow'),0o755) # read/execute permissions for all

def adjust_clm_files(dir_setup,dir_run,settings_clm):
    with open(os.path.join(dir_setup,'namelists/lnd.stdin'), 'r') as file :
      filedata = file.read()

    time_seconds_start = (settings_clm['datetime_start'].hour*60*60 + 
                          settings_clm['datetime_start'].minute*60 + 
                          settings_clm['datetime_start'].second)
    
    filedata = filedata.replace('__time_totalrun__', '%i'% (settings_clm['t_total'].total_seconds()/(60*60*24)) ) #in days
    filedata = filedata.replace('__time_clm_dump__', '%i'% (settings_clm['t_dump'].total_seconds()/(60*60)) ) #in hours
    filedata = filedata.replace('__time_couple__', '%i'% settings_clm['t_couple'].total_seconds() )  #in seconds  
    filedata = filedata.replace('__date_start__', '%s'% str(settings_clm['datetime_start'].date()).replace('-','') )
    filedata = filedata.replace('__time_start__', '%i'% time_seconds_start )    
    filedata = filedata.replace('__dir_forcing__', '%s'% settings_clm['dir_forcing'])
    filedata = filedata.replace('__clm_nx__', '%s'% settings_clm['nx'])
    filedata = filedata.replace('__clm_ny__', '%s'% settings_clm['ny'])
    filedata = filedata.replace('__file_restart__', '%s'% settings_clm['file_restart'])
    
    with open(os.path.join(dir_run,'lnd.stdin'), 'w') as file:
      file.write(filedata)
    os.chmod(os.path.join(dir_run,'lnd.stdin'),0o755) # read/execute permissions for all

def adjust_parflow_files(dir_setup,dir_run,dir_build,dir_bin,settings_pfl):
    for file_ in ['input_pf/ascii2pfb_slopes.tcl','input_pf/ascii2pfb_SoilInd.tcl','namelists/coup_oas.tcl']:

        with open( os.path.join(dir_setup,file_) , 'r') as file :
          filedata = file.read()

        filedata = filedata.replace('__nprocx_pfl_bldsva__', '%i'%settings_pfl['n_proc_pfl_x'])
        filedata = filedata.replace('__nprocy_pfl_bldsva__', '%i'%settings_pfl['n_proc_pfl_y'])
        filedata = filedata.replace('__pfl_dx__', settings_pfl['dx'])
        filedata = filedata.replace('__pfl_dy__', settings_pfl['dy'])
        filedata = filedata.replace('__pfl_nx__', '%i'%settings_pfl['nx'])
        filedata = filedata.replace('__pfl_ny__', '%i'%settings_pfl['ny'])

        filedata = filedata.replace('__dir_bin__', '%s'%dir_bin)
        filedata = filedata.replace('__dir_pfl__', '%s'%os.path.join(dir_build,'parflow') )
        filedata = filedata.replace('__dir_pfin__', '%s'% os.path.join(dir_setup,'input_pf') )
        filedata = filedata.replace('__dir_run__', '%s'% dir_run )

        filedata = filedata.replace('__time_totalrun__', '%i'% (settings_pfl['t_total'].total_seconds()/(60*60)) )
        filedata = filedata.replace('__time_pfl_dump__', '%.1f'% (settings_pfl['t_dump'].total_seconds()/(60*60)) )
        filedata = filedata.replace('__time_couple__', '%.2f'% (settings_pfl['t_couple'].total_seconds()/(60*60)) )   
        filedata = filedata.replace('__filename_pfl_out__', settings_pfl['filename_pfl_out'] )

        filedata = filedata.replace('__icpres_type__', '%s'%settings_pfl['icpres_type'])
        filedata = filedata.replace('__geom_icpres_valorfile__', '%s'%settings_pfl['geom_icpres_valorfile'])
        filedata = filedata.replace('__geom_icpres_val__', '%s'%settings_pfl['geom_icpres_val'])
        
        
        with open(os.path.join(dir_run,os.path.basename(file_)), 'w') as file:
          file.write(filedata)
        os.chmod(os.path.join(dir_run,os.path.basename(file_)),0o755)

def make_parflow_executable(dir_run,dir_build):
    str_cmd = '''
    source %s
    tclsh %s
    tclsh %s
    tclsh %s
    ''' % (os.path.join(dir_build,'bldsva/machines/JUWELS/loadenvs.Intel'),
           os.path.join(dir_run,'ascii2pfb_slopes.tcl'),
           os.path.join(dir_run,'ascii2pfb_SoilInd.tcl'),
           os.path.join(dir_run,'coup_oas.tcl'))

    os.system(str_cmd)    

def adjust_oasis_files(dir_setup,dir_run,settings_clm,settings_pfl):
    with open(os.path.join(dir_setup,'namelists/namcouple_pfl_clm'), 'r') as file :
      filedata = file.read()

    filedata = filedata.replace('__runTime__', '%i'% settings_clm['t_total'].total_seconds())

    filedata = filedata.replace('__cplfreq__', '%i'% settings_clm['t_couple'].total_seconds())
    filedata = filedata.replace('__ngpflx__', '%i'% settings_clm['nx'])
    filedata = filedata.replace('__ngpfly__', '%i'% settings_clm['ny'])
    filedata = filedata.replace('__ngclmx__', '%i'% (settings_clm['nx']*settings_clm['ny'])) #the clm grid is 'flattened' in oasis
    filedata = filedata.replace('__ngclmy__', '%i'% 1)

    file_out = os.path.join(dir_run,'namcouple')
    # Write the file out again
    with open(file_out, 'w') as file:
      file.write(filedata)

    os.chmod(file_out,0o755) # read/execute permissions for all
    
def copy_oasis_files(dir_setup,dir_run):
    shutil.copy( os.path.join(dir_setup,'input_oas/rmp_gclm_to_gpfl_DISTWGT.nc'), dir_run )
    shutil.copy( os.path.join(dir_setup,'input_oas/rmp_gpfl_to_gclm_BILINEAR.nc'), dir_run )
    shutil.copy( os.path.join(dir_setup,'input_oas/clmgrid.nc'), dir_run )
    
def adjust_run_files(dir_setup,dir_run,settings_clm,settings_pfl,settings_sbatch):

    n_proc_pfl = settings_pfl['n_proc_pfl_x']*settings_pfl['n_proc_pfl_y']*settings_pfl['n_proc_pfl_z']
    n_proc_clm = settings_clm['n_proc_clm']
    proc_pfl_0 = 0
    proc_pfl_n = n_proc_pfl - 1 
    proc_clm_0 = n_proc_pfl
    proc_clm_n = n_proc_pfl + n_proc_clm - 1

    with open( os.path.join(dir_setup,'namelists/slm_multiprog_mapping.conf') , 'r') as file :
      filedata = file.read()

    filedata = filedata.replace('__pfl0__', '%i'%proc_pfl_0)
    filedata = filedata.replace('__pfln__', '%i'%proc_pfl_n)
    filedata = filedata.replace('__clm0__', '%i'%proc_clm_0)
    filedata = filedata.replace('__clmn__', '%i'%proc_clm_n)
    filedata = filedata.replace('__filename_pfl__', '%s'%settings_pfl['filename_pfl_out'])

    with open(os.path.join(dir_run,'slm_multiprog_mapping.conf'), 'w') as file:
      file.write(filedata)
    os.chmod(os.path.join(dir_run,'slm_multiprog_mapping.conf'),0o755)

    
    n_nodes = int(np.ceil( (n_proc_pfl + n_proc_clm) / 48 )) # 48 cores per node on JUWEL
    
    with open( os.path.join(dir_setup,'namelists/tsmp_slm_run.bsh') , 'r') as file :
      filedata = file.read()

    filedata = filedata.replace('__ntasks__', '%i'%(n_proc_pfl + n_proc_clm))
    filedata = filedata.replace('__nnodes__', '%i'%n_nodes)
    filedata = filedata.replace('__stime__', settings_sbatch['sbatch_time'])
    filedata = filedata.replace('__spart__', settings_sbatch['sbatch_partition'])
    filedata = filedata.replace('__sacc__', settings_sbatch['sbatch_account'])

    with open(os.path.join(dir_run,'tsmp_slm_run.bsh'), 'w') as file:
      file.write(filedata)
    os.chmod(os.path.join(dir_run,'tsmp_slm_run.bsh'),0o755)
    
def start_run(dir_run):
    # os.chdir(dir_run)
    # sbatch_file = os.path.join(dir_run,'tsmp_slm_run.bsh')
    sbatch_file = 'tsmp_slm_run.bsh'
    # os.system('sbatch %s' % sbatch_file)
    
    str_cmd = '''
    source %s
    sbatch %s
    ''' % (os.path.join(dir_build,'bldsva/machines/JUWELS/loadenvs.Intel'),
           sbatch_file)
    os.system(str_cmd)    
    
def wait_for_run(dir_run,settings_sbatch):
    while not os.path.exists(os.path.join(dir_run,'ready.txt')):
        print('Still running...')
        time.sleep(settings_sbatch['sbatch_check_sec'])
        
def move_and_link(to_link,folder_store):
    # move file to folder_store, and keep a symbolic link
    if type(to_link)==list:
        for file in to_link:
            if not os.path.islink(file):
                shutil.move( file, folder_store)

                # link files to rundir
                file_ln = os.path.join(folder_store,os.path.basename(file) )
                os.symlink(file_ln, file ) #verify this

    elif type(to_link)==str: # in this case 'to_link' is a single file or a folder
        if not os.path.islink(to_link):
            shutil.move( to_link, folder_store)
            print('File/dir moved to storage: %s' % folder_store)

            # link files/folder back into the rundir
            file_ln = os.path.join(folder_store,os.path.basename(to_link) )
            print('Linking file/dir from %s to %s' % (file_ln,to_link))
            os.symlink(file_ln, to_link )   
             
    else:
        raise RuntimeError
        
        
date_start = datetime(2009,1,1,12,0,0)
date_end = datetime(2010,1,1,12,0,0)
# date_start = datetime(2010,1,7,12,0,0)
# date_end = datetime(2011,1,11,12,0,0)

freq_restart = 'AS'#'7d'#'AS','MS' # AS = annual, start of year (see pandas date_range freq options)
freq_output = 'MS'#'1d'#'7d' # see pandas date_range freq options
time_couple = timedelta(seconds=900) # coupling, don't change this - or check coup_oas.tcl carefully (e.g. baseunit) - pfl units are in hours
nx = 111 #111,222,444
ny = 108 #108,216,432

dir_forcing = '/p/scratch/cjibg36/kaandorp2/data/ERA5_EUR-44_CLM' #folder containing CLM forcing files

dir_setup = '/p/project/cjibg36/kaandorp2/TSMP_patched/tsmp_cordex_spinup_%ix%i' % (nx,ny) #folder in which the case will be run
dir_build = '/p/project/cjibg36/kaandorp2/TSMP_patched/' #required for parflow files
dir_binaries = '/p/project/cjibg36/kaandorp2/TSMP_patched/bin/JUWELS_3.1.0MCT_clm-pfl' #folder from which parflow/clm binaries are to be copied
dir_store = '/p/scratch/cjibg36/kaandorp2/TSMP_results/TSMP_patched/tsmp_cordex_spinup_%ix%i' % (nx,ny) #files are moved here after the run is finished
dir_template = '/p/project/cjibg36/kaandorp2/TSMP_setups/setup_tsmp_cordex_%ix%i/' % (nx,ny) #only necessary for a new run

date_restarts = pd.date_range(date_start,date_end,freq=freq_restart) #Restart annualy at the start of the year
date_results = pd.date_range(date_start,date_end,freq=freq_output) #output is written every 7 days

spinup = 10 # integer or set to False
init_restart = False #initial restart files are available?
IC_file_CLM = '/p/project/cjibg36/kaandorp2/TSMP_patched/tsmp_cordex_111x108/run_20100101-20110107/clm.clm2.r.2011-01-07-43200.nc'
IC_file_ParFlow = '/p/project/cjibg36/kaandorp2/TSMP_patched/tsmp_cordex_111x108/run_20100101-20110107/cordex111x108_20100101-20110107.out.00053.nc'

#---Some options for n_nodes(n_cores)=1(48),2(96),3(144),4(192),6(288)
n_proc_pfl_x = 11 #6,9,11,12,15
n_proc_pfl_y = 11 #6,9,11,12,15
n_proc_pfl_z = 1
n_proc_clm = 23 #12,15,23,48,63
sbatch_account = 'jibg36'
sbatch_partition = 'batch' #batch
sbatch_time = '1-00:00:00' #1-00:00:00 
sbatch_check_sec = 60*5 #check every n seconds if the simulation is done

# EU-cordex specific: domain extend
x_tot = 5550000
y_tot = 5400000

time_dump = date_results[1]-date_results[0]

settings_clm = {'t_dump':time_dump,
                't_couple':time_couple,
                'dir_forcing':dir_forcing,
                'nx':nx,
                'ny':ny,
                'n_proc_clm':n_proc_clm}

settings_pfl = {'t_dump':time_dump,
                't_couple':time_couple,
                'nx':nx,
                'ny':ny,
                'dx':('%.1f' % (x_tot/nx)),
                'dy':('%.1f' % (y_tot/ny)),
                'n_proc_pfl_x':n_proc_pfl_x,
                'n_proc_pfl_y':n_proc_pfl_y,
                'n_proc_pfl_z':n_proc_pfl_z}
               
settings_sbatch = {'sbatch_account':sbatch_account,
                  'sbatch_partition':sbatch_partition,
                  'sbatch_time':sbatch_time,
                  'sbatch_check_sec':sbatch_check_sec}               

# Create a list which contains for every restart time period (e.g. 1 year) all dates for which output is written (i.e. that can be used for restart runs)
date_results_binned = bin_dates_by_restart_dates(date_results,date_restarts,spinup=spinup)

    
'''
 1) Copy the folder template to the setup location if the destination does not exist
'''
if not os.path.exists(dir_setup):
    print('Copying folder template from %s to %s' % (dir_template,dir_setup) )
    shutil.copytree(dir_template,dir_setup)
else:
    print('Continuing simulation in %s' % dir_setup)
os.chdir(dir_setup)
        
        
'''
 2) Loop through the restart dates, if the folder does not exist create and setup run

'''
for i1,date_list in enumerate(date_results_binned):
    
    if i1==0 and not init_restart:
        flag_restart = False
        settings_clm['file_restart'] = ''
        settings_pfl['icpres_type'] = 'HydroStaticPatch'
        settings_pfl['geom_icpres_valorfile'] = 'Value'
        settings_pfl['geom_icpres_val'] = '-0.2'
    elif i1 ==0 and init_restart: #provide a IC file from another run
        print('Restarting run from %s and %s' % (IC_file_CLM,IC_file_ParFlow) )
        flag_restart = True
        settings_clm['file_restart'] = '%s'%IC_file_CLM #TODO
        settings_pfl['icpres_type'] = 'NCFile'
        settings_pfl['geom_icpres_valorfile'] = 'FileName'
        settings_pfl['geom_icpres_val'] = '"%s"'%IC_file_ParFlow #TODO
    else: #base the initial conditions on restart file from previous date
        flag_restart = True
        date_list_prev = date_results_binned[i1-1]
        
        date_start_prev = date_list_prev[0]
        date_end_prev = date_list_prev[-1]
        str_date_prev = str(date_start_prev.date()).replace('-','') + '-' + str(date_end_prev.date()).replace('-','')
        if spinup:
            str_spinup_prev = '%3.3i_' % (i1-1)
        else:
            str_spinup_prev = ''
        dir_run_prev = os.path.join(dir_setup,'run_%s%s' % (str_spinup_prev,str_date_prev) )
    
        files_clm_prev = sorted(glob(os.path.join(dir_run_prev,'clm.clm2.r.*.nc') ))
        assert len(files_clm_prev) == (len(date_list_prev)-1), 'Check if previous run crashed, available CLM restart files: %s' % files_clm_prev
        settings_pfl['icpres_type'] = 'NCFile'
        settings_pfl['geom_icpres_valorfile'] = 'FileName'    
        settings_clm['file_restart'] = files_clm_prev[-1]
        settings_pfl['geom_icpres_val'] = sorted(glob(os.path.join(dir_run_prev,'cordex%ix%i_%s.out.0*' % (nx,ny,str_date_prev))))[-1]
        
        
    date_start_sim = date_list[0]
    date_end_sim = date_list[-1]

    str_date = str(date_start_sim.date()).replace('-','') + '-' + str(date_end_sim.date()).replace('-','')
    if spinup:
        assert type(spinup) == int, 'Spinup needs to be False or an integer' 
        print('Running in spinup mode! Repeating %i times' % spinup)
        str_spinup = '%3.3i_' % i1
    else:
        str_spinup = ''
    dir_run = os.path.join(dir_setup,'run_%s%s' % (str_spinup,str_date) )
    if not os.path.exists(dir_run):
        print('Preparing simulation in %s' % dir_run )
        os.mkdir(dir_run)

        copy_binaries(dir_binaries,dir_run)
        
        settings_clm['t_total'] = date_end_sim-date_start_sim
        settings_clm['datetime_start'] = date_start_sim
        adjust_clm_files(dir_setup,dir_run,settings_clm)
        
        settings_pfl['t_total'] = date_end_sim-date_start_sim
        settings_pfl['filename_pfl_out'] = 'cordex%ix%i_%s' % (nx,ny,str_date)
        adjust_parflow_files(dir_setup,dir_run,dir_build,dir_binaries,settings_pfl)
        make_parflow_executable(dir_run,dir_build)
        
        adjust_oasis_files(dir_setup,dir_run,settings_clm,settings_pfl)
        copy_oasis_files(dir_setup,dir_run)
        
        adjust_run_files(dir_setup,dir_run,settings_clm,settings_pfl,settings_sbatch)
    
        os.chdir(dir_run)
        start_run(dir_run)
        wait_for_run(dir_run,settings_sbatch)
        os.chdir(dir_setup)
        
        if not os.path.exists(dir_store):
            print('Creating dir: %s' % dir_store)
            os.makedirs(dir_store)
        print('Moving files to storage: %s' % dir_store)
        move_and_link(dir_run,dir_store)
        
            
    else:
        print('%s exists, continuing...' % dir_run) 
    