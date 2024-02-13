from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr
import os
import shutil
import time
from glob import glob

from settings import settings_run,settings_clm,settings_pfl,settings_sbatch, date_results_binned

'''
v2: test adjusting Ks tensor value
'''

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
    filedata = filedata.replace('__dir_clmin__', '%s'% os.path.join(dir_setup,'input_clm') )

    with open(os.path.join(dir_run,'lnd.stdin'), 'w') as file:
      file.write(filedata)
    os.chmod(os.path.join(dir_run,'lnd.stdin'),0o755) # read/execute permissions for all

def adjust_parflow_files(dir_setup,dir_run,dir_build,dir_bin,dir_real,settings_pfl):
    for file_ in ['input_pf/ascii2pfb_slopes.tcl','input_pf/ascii2pfb_SoilInd.tcl','input_pf/ascii2pfb_Ks.tcl','namelists/coup_oas.tcl']:

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
        filedata = filedata.replace('__dir_real__', '%s'%dir_real)
        
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
    tclsh %s
    ''' % (os.path.join(dir_build,'bldsva/machines/JUWELS/loadenvs.Intel'),
           os.path.join(dir_run,'ascii2pfb_slopes.tcl'),
           os.path.join(dir_run,'ascii2pfb_SoilInd.tcl'),
           os.path.join(dir_run,'ascii2pfb_Ks.tcl'),
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
    
def start_run(dir_build):
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
        print('Still running...',flush=True)
        time.sleep(settings_sbatch['sbatch_check_sec'])
        
def move_and_link(to_link,folder_store,delete_existing=True):
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
            if delete_existing and os.path.exists(folder_store):
                shutil.rmtree(folder_store)
            
            shutil.move( to_link, folder_store)
            print('File/dir moved to storage: %s' % folder_store)

            # link files/folder back into the rundir
            file_ln = os.path.join(folder_store,os.path.basename(to_link) )
            print('Linking file/dir from %s to %s' % (file_ln,to_link))
            os.symlink(file_ln, to_link )   
             
    else:
        raise RuntimeError
    
def remove_misc_files(dir_run):
    # [os.remove(file) for file in glob(os.path.join(dir_run,'CLMSAT*.nc'))]
    # Keep all output files
    retain = ['mpiMPMD*','cordex*.out.0*','clm.clm2.h0*.nc','clm.clm2.r*.nc','coup_oas*','lnd*','ready*']
    files_retain = []
    for retain_ in retain:
        files_retain.extend(glob(os.path.join(dir_run,retain_)))

    # Keep the last CLM restart file
    # files_retain.append(sorted(glob(os.path.join(dir_run,'clm.clm2.r*.nc')))[-1])
    # files_retain.append(sorted(glob(os.path.join(dir_run,'clm.clm2.r*.nc'))) )
    
    # Check for all files if necessary to retain, if not delete
    files = os.listdir(dir_run)
    for file_ in files:
        if os.path.join(dir_run,file_) not in files_retain:
            # print(os.path.join(dir_run,file_))
            os.remove(os.path.join(dir_run,file_))
            
    
def setup_submit_wait(i_real,settings_run,settings_clm,settings_pfl,settings_sbatch,date_results_iter):
    
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    for i1,date_list in enumerate(date_results_iter):

        if i1==0 and not settings_run['init_restart']: #'Cold' run
            flag_restart = False
            settings_clm['file_restart'] = ''
            settings_pfl['icpres_type'] = 'HydroStaticPatch'
            settings_pfl['geom_icpres_valorfile'] = 'Value'
            settings_pfl['geom_icpres_val'] = '-2.0'
        elif i1 ==0 and settings_run['init_restart']: #provide a IC file from another run
            print('Restarting run from %s and %s' % (settings_clm['IC_file'],settings_pfl['IC_file']) )
            flag_restart = True
            settings_clm['file_restart'] = '%s' % settings_clm['IC_file'] 
            settings_pfl['icpres_type'] = 'NCFile'
            settings_pfl['geom_icpres_valorfile'] = 'FileName'
            settings_pfl['geom_icpres_val'] = '"%s"' % settings_pfl['IC_file']
        else: #base the initial conditions on restart file from previous date
            flag_restart = True
            date_list_prev = date_results_iter[i1-1]

            date_start_prev = date_list_prev[0]
            date_end_prev = date_list_prev[-1]
            str_date_prev = str(date_start_prev.date()).replace('-','') + '-' + str(date_end_prev.date()).replace('-','')
            if settings_run['spinup']:
                str_spinup_prev = '%3.3i_' % (i1-1)
            else:
                str_spinup_prev = ''
            dir_run_prev = os.path.join(dir_real,'run_%s%s' % (str_spinup_prev,str_date_prev) )

            files_clm_restart = sorted(glob(os.path.join(dir_run_prev,'clm.clm2.r.*.nc') ))
            # if settings_run['spinup']:
            assert str(date_end_prev.date()) in files_clm_restart[-1], 'Check if previous run crashed, last date: %s, restart file: %s'%(str(date_end_prev.date()),files_clm_restart[-1])
            
            # files_clm_prev = sorted(glob(os.path.join(dir_run_prev,'clm.clm2.h0.*.nc') ))            
            # assert len(files_clm_prev) == (len(date_list_prev)-1), 'Check if previous run crashed, available CLM output files: %s, previous dates: %s' % (files_clm_prev, date_list_prev)
            settings_pfl['icpres_type'] = 'NCFile'
            settings_pfl['geom_icpres_valorfile'] = 'FileName'    
            settings_clm['file_restart'] = files_clm_restart[-1]
            settings_pfl['geom_icpres_val'] = sorted(glob(os.path.join(dir_run_prev,
                                                                       'cordex%ix%i_%s.out.0*' % (settings_pfl['nx'],settings_pfl['ny'],str_date_prev))))[-1]
            print('Restarting CLM from %s' % files_clm_restart[-1])
            

        date_start_sim = date_list[0]
        date_end_sim = date_list[-1]

        ## Define the run directory, as 'run_{start_date}-{end_date}', or 'run_{integer}_{start_date}-{end_date}' in the case of a spinup with the same dates repeated
        str_date = str(date_start_sim.date()).replace('-','') + '-' + str(date_end_sim.date()).replace('-','')
        if settings_run['spinup']:
            assert type(settings_run['spinup']) == int, 'Spinup needs to be False or an integer' 
            print('Running in spinup mode! Repeating %i times' % settings_run['spinup'])
            str_spinup = '%3.3i_' % i1
        else:
            str_spinup = ''
        dir_run = os.path.join(dir_real,'run_%s%s' % (str_spinup,str_date) )

        ## Main loop: if the given run directory does not exist prepare and submit the run
        if not os.path.exists(dir_run):
            print('Preparing simulation in %s' % dir_run )
            os.mkdir(dir_run)

            copy_binaries(settings_run['dir_binaries'],dir_run)

            settings_clm['t_total'] = date_end_sim-date_start_sim
            settings_clm['datetime_start'] = date_start_sim
            adjust_clm_files(settings_run['dir_setup'],dir_run,settings_clm)

            settings_pfl['t_total'] = date_end_sim-date_start_sim
            settings_pfl['filename_pfl_out'] = 'cordex%ix%i_%s' % (settings_pfl['nx'],settings_pfl['ny'],str_date)
            adjust_parflow_files(settings_run['dir_setup'],dir_run,
                                 settings_run['dir_build'],settings_run['dir_binaries'],
                                 dir_real,settings_pfl)
            make_parflow_executable(dir_run,settings_run['dir_build'])

            adjust_oasis_files(settings_run['dir_setup'],dir_run,settings_clm,settings_pfl)
            copy_oasis_files(settings_run['dir_setup'],dir_run)

            adjust_run_files(settings_run['dir_setup'],dir_run,settings_clm,settings_pfl,settings_sbatch)

            os.chdir(dir_run)
            start_run(settings_run['dir_build'])
            ## in tsmp_slm_run.bsh, a file ready.txt is written at the end of the run: wait for this file
            wait_for_run(dir_run,settings_sbatch) 
            os.chdir(settings_run['dir_setup'])
          
            remove_misc_files(dir_run)

            # ## Last step: move the run directory to storage (scratch), and keep a link
            # if not os.path.exists(settings_run['dir_store']):
            #     print('Creating dir: %s' % settings_run['dir_store'])
            #     os.makedirs(settings_run['dir_store'])
            # print('Moving files to storage: %s' % settings_run['dir_store'])
            # move_and_link(dir_run,settings_run['dir_store'])


        else:
            print('%s exists, continuing...' % dir_run) 

if __name__ == '__main__':
    
    settings_run['dir_real'] = '/p/scratch/cjibg36/kaandorp2/TSMP_results/TSMP_patched/DA_tsmp_cordex_111x108/20190102-20191231/i000/R000'
    date_results_iter = date_results_binned[0]
    
    setup_submit_wait(settings_run,settings_clm,settings_pfl,settings_sbatch,date_results_iter)