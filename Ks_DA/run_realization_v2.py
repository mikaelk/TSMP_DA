from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr
import os
import shutil
import time
from glob import glob
import copy

from settings import settings_run,settings_clm,settings_pfl,settings_sbatch, date_results_binned

'''
v2:   test adjusting Ks tensor value
v2.1: porosity option added
v2.2: should be compatible with all parameters enabled/disabled
v2.3: compatible with eCLM
'''

def copy_binaries(settings_run,dir_run):
   
    dir_binaries = settings_run['dir_binaries']
    print('Copy binaries from %s' % dir_binaries)

    if 'eCLM' in settings_run['models']:
        shutil.copyfile(os.path.join(dir_binaries,'eclm.exe'), os.path.join(dir_run,'eclm.exe'))
        os.chmod(os.path.join(dir_run,'eclm.exe'),0o755) # read/execute permissions for all

    else:
        shutil.copyfile(os.path.join(dir_binaries,'clm'), os.path.join(dir_run,'clm'))
        os.chmod(os.path.join(dir_run,'clm'),0o755) # read/execute permissions for all

    if 'PFL' in settings_run['models']:
        shutil.copyfile(os.path.join(dir_binaries,'parflow'), os.path.join(dir_run,'parflow'))
        os.chmod(os.path.join(dir_run,'parflow'),0o755) # read/execute permissions for all

def adjust_clm_files(dir_setup,dir_run,settings_clm):
    with open(os.path.join(dir_setup,'namelists/lnd.stdin'), 'r') as file : #replaced namelists
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

def adjust_eclm_files(settings_run,dir_run,dir_real,settings_clm):
    dir_setup = settings_run['dir_setup']
    files_modelio = glob(os.path.join(dir_setup,'input_clm/*modelio.nml') )
    files_stream_t = glob(os.path.join(dir_setup,'input_clm/*streams*') )

    if 'PFL' in settings_run['models']: #TODO: check why different domain/surface files are required when including ParFlow...
        filename_domain = 'domain.lnd.EUR-11_EUR-11.230216_landfrac.nc' 
        filename_landfrac = 'domain.lnd.EUR-11_EUR-11.230216_landfrac.nc'
        # filename_surf = 'surfdata_EUR-11_hist_16pfts_Irrig_CMIP6_simyr2000_c230808_GLC2000.nc' #this is not working yet in the coupled setup (?)
        filename_surf = 'surfdata_clm3.nc'
        # filename_landfrac = 'domain.lnd.EUR-11_EUR-11.230216_landfrac.nc' #mask is 0 everywhere..?

    else: #working eCLM setup
        filename_domain = 'domain.lnd.EUR-11_EUR-11.230216_mask.nc'
        filename_landfrac = 'domain.lnd.EUR-11_EUR-11.230216_mask.nc'
        filename_surf = 'surfdata_EUR-11_hist_16pfts_Irrig_CMIP6_simyr2000_c230808_GLC2000.nc'
        # filename_domain = 'domain.lnd.EUR-11_EUR-11.230216_landfrac.nc'
        # filename_landfrac = 'domain.lnd.EUR-11_EUR-11.230216_landfrac.nc'
        # filename_surf = 'surfdata_EUR-11_hist_16pfts_Irrig_CMIP6_simyr2000_c230808_GLC2000.nc'
    
    if 'sandfrac_anom' in settings_clm['param_names'] or 'clayfrac_anom' in settings_clm['param_names'] or 'orgfrac_anom' in settings_clm['param_names']:
        dir_surf = dir_real
    else:
        dir_surf = os.path.join(dir_setup,'input_clm')

    # Important: define all parameters that can be assimilated by adjusting the parameter file
    all_clm_parameters = ['fff','orgmax','medlyn_slope','medlyn_intercept',
                  'b_slope', 'b_intercept', 'log_psis_slope', 'log_psis_intercept', 
                  'log_ks_slope', 'log_ks_intercept', 'thetas_slope', 'thetas_intercept',
                         'om_hydraulic','om_thermal','h2o_canopy_max','kmax', 'mineral_hydraulic']
    filename_param = 'clm5_params.c171117.nc'
    if any(i in all_clm_parameters for i in settings_clm['param_names']):
        file_param = os.path.join(dir_real,filename_param)
        if not os.path.exists(file_param):
            shutil.copyfile(os.path.join(dir_setup,'input_clm/%s'%filename_param), file_param )
    else:
        file_param = os.path.join(dir_setup,'input_clm/%s'%filename_param)
    
    filename_snowage = 'snicar_drdt_bst_fit_60_c070416.nc'
    filename_snowopt = 'snicar_optics_5bnd_c090915.nc'
    filename_urb = 'CLM50_tbuildmax_Oleson_2016_0.9x1.25_simyr1849-2106_c160923.nc'

    filename_ndep = 'fndep_clm_hist_b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.ensmean_1849-2015_monthly_0.9x1.25_c180926.nc' #BCG only?
    filename_ch4 = 'finundated_inversiondata_0.9x1.25_c170706.nc' #BCG only?
    filename_popd = 'clmforc.Li_2017_HYDEv3.2_CMIP6_hdm_0.5x0.5_AVHRR_simyr1850-2016_c180202.nc' #not used
    filename_light = 'clmforc.Li_2012_climo1995-2011.T62.lnfm_Total_c140423.nc' #not used

    ### Adjust all *modelio.nml namelist files
    index_log = '0000000'
    os.mkdir(os.path.join(dir_run,'log'))
    os.makedirs(os.path.join(dir_run,'timing/checkpoints'))
        
    dir_log = os.path.join(dir_run,'log')

    for file_ in files_modelio:

        with open(file_, 'r') as file :
          filedata = file.read()

        filedata = filedata.replace('__dir_log__', '%s'% dir_log ) 
        filedata = filedata.replace('__index__', '%s'% index_log ) 

        with open(os.path.join(dir_run,os.path.basename(file_)), 'w') as file:
          file.write(filedata)
        os.chmod(os.path.join(dir_run,os.path.basename(file_)),0o755) # read/execute permissions for all

    ### Adjust all stream files
    date_min = datetime(settings_clm['datetime_start'].year,settings_clm['datetime_start'].month,1)
    date_max = datetime(settings_clm['datetime_end'].year,settings_clm['datetime_end'].month,1)
    dates_files = pd.date_range(date_min,date_max,freq='MS')

    list_files = ''
    for i1,date_ in enumerate(dates_files):
        list_files += '      %4.4i-%2.2i.nc' % (date_.year,date_.month)
        if i1 < len(dates_files)-1:
            list_files += '\n'

    for file_ in files_stream_t:
        with open(file_, 'r') as file :
          filedata = file.read()

        filedata = filedata.replace('__path_in_clm__', '%s'% os.path.join(dir_setup,'input_clm') )
        filedata = filedata.replace('__path_common_clm__', '%s'% settings_clm['dir_common_eclm'] )
        filedata = filedata.replace('__path_forcing__', '%s'% settings_run['dir_forcing'] )
        filedata = filedata.replace('__list_filenames__', '%s'% list_files )
        filedata = filedata.replace('__file_domain__', '%s'% filename_domain )

        with open(os.path.join(dir_run,os.path.basename(file_)), 'w') as file:
          file.write(filedata)
        os.chmod(os.path.join(dir_run,os.path.basename(file_)),0o755) # read/execute permissions for all
        
        
    ### Adjust datm_in
    file_ = os.path.join(dir_setup,'input_clm/datm_in')
    with open(file_, 'r') as file :
      filedata = file.read()

    filedata = filedata.replace('__file_domain__', '%s'% os.path.join(dir_setup,'input_clm/%s'%filename_domain) )
    filedata = filedata.replace('__year_start__', '%s'% date_min.year )
    filedata = filedata.replace('__year_end__', '%s'% date_max.year )

    with open(os.path.join(dir_run,os.path.basename(file_)), 'w') as file:
      file.write(filedata)
    os.chmod(os.path.join(dir_run,os.path.basename(file_)),0o755) # read/execute permissions for all

    
    ### drv_flds_in 
    file_ = os.path.join(dir_setup,'input_clm/drv_flds_in')
    with open(file_, 'r') as file :
      filedata = file.read()

    filedata = filedata.replace('__file_megan__', '%s'% os.path.join(settings_clm['dir_common_eclm'],'megan21_emis_factors_78pft_c20161108.nc') )

    with open(os.path.join(dir_run,os.path.basename(file_)), 'w') as file:
      file.write(filedata)
    os.chmod(os.path.join(dir_run,os.path.basename(file_)),0o755) # read/execute permissions for all
    
    ### drv_in 
    time_seconds_start = (settings_clm['datetime_start'].hour*60*60 + 
                          settings_clm['datetime_start'].minute*60 + 
                          settings_clm['datetime_start'].second)

    time_seconds_end = (settings_clm['datetime_end'].hour*60*60 + 
                        settings_clm['datetime_end'].minute*60 + 
                        settings_clm['datetime_end'].second)

    file_ = os.path.join(dir_setup,'input_clm/drv_in')
    with open(file_, 'r') as file :
      filedata = file.read()

    filedata = filedata.replace('__nclm_proc__', '%i'% settings_clm['n_proc_clm'] )
    filedata = filedata.replace('__cplfrq__', '%i'% settings_clm['t_couple'].total_seconds() )

    filedata = filedata.replace('__date_start__', '%s'% str(settings_clm['datetime_start'].date()).replace('-','') )
    filedata = filedata.replace('__time_start__', '%i'% time_seconds_start )  
    filedata = filedata.replace('__date_end__', '%s'% str(settings_clm['datetime_end'].date()).replace('-','') )
    filedata = filedata.replace('__time_end__', '%i'% time_seconds_end ) 
    filedata = filedata.replace('__date_restart__', '%s'% str(settings_clm['datetime_end'].date()).replace('-','') )


    with open(os.path.join(dir_run,os.path.basename(file_)), 'w') as file:
      file.write(filedata)
    os.chmod(os.path.join(dir_run,os.path.basename(file_)),0o755) # read/execute permissions for all
    
    ### lnd_in
    file_ = os.path.join(dir_setup,'input_clm/lnd_in')
    with open(file_, 'r') as file :
      filedata = file.read()

    filedata = filedata.replace('__cplfrq__', '%i'% settings_clm['t_couple'].total_seconds() )
    filedata = filedata.replace('__time_clm_dump__', '%i'% (settings_clm['t_dump'].total_seconds()/(60*60)) ) #in hours

    filedata = filedata.replace('__vars_dump__', '%s'% str(settings_clm['vars_dump']).split('[')[1].split(']')[0] ) 
    filedata = filedata.replace('__vars_dump2__', '%s'% str(settings_clm['vars_dump2']).split('[')[1].split(']')[0] )
    filedata = filedata.replace('__time_clm_dump2__', '%i'% (settings_clm['t_dump2'].total_seconds()/(60*60)) ) #in hours
    
    # files in the input_clm folder
    filedata = filedata.replace('__file_domain__', '%s'% os.path.join(dir_setup,'input_clm/%s'%filename_landfrac) )
    filedata = filedata.replace('__file_surf__', '%s'% os.path.join(dir_surf,filename_surf))
    filedata = filedata.replace('__file_restart__', '%s'% settings_clm['file_restart'])
    filedata = filedata.replace('__file_param__', '%s'% file_param)
    
    # files in the common data folder
    filedata = filedata.replace('__file_snowage__', '%s'% os.path.join(settings_clm['dir_common_eclm'],filename_snowage))
    filedata = filedata.replace('__file_snowopt__', '%s'% os.path.join(settings_clm['dir_common_eclm'],filename_snowopt))
    filedata = filedata.replace('__file_ndep__', '%s'% os.path.join(settings_clm['dir_common_eclm'],filename_ndep))
    filedata = filedata.replace('__file_popd__', '%s'% os.path.join(settings_clm['dir_common_eclm'],filename_popd))
    filedata = filedata.replace('__file_urb__', '%s'% os.path.join(settings_clm['dir_common_eclm'],filename_urb))
    filedata = filedata.replace('__file_light__', '%s'% os.path.join(settings_clm['dir_common_eclm'],filename_light))
    filedata = filedata.replace('__file_ch4__', '%s'% os.path.join(settings_clm['dir_common_eclm'],filename_ch4))
    if settings_clm['init_interp'] == True:
        filedata = filedata.replace('__init_interp__', '%s'% '.true.')
    else:
        filedata = filedata.replace('__init_interp__', '%s'% '.false.')    
    
    with open(os.path.join(dir_run,os.path.basename(file_)), 'w') as file:
      file.write(filedata)
    os.chmod(os.path.join(dir_run,os.path.basename(file_)),0o755) # read/execute permissions for all

    ## copy mosart namelist
    shutil.copyfile(os.path.join(dir_setup,'input_clm/mosart_in'), os.path.join(dir_run,'mosart_in'))

    
def adjust_parflow_files(dir_setup,dir_run,dir_build,dir_bin,dir_real,settings_pfl):
    # files which need to be adjusted
    files_parflow = ['coup_oas.tcl',
                     'ascii2pfb_SoilInd.tcl',
                     'ascii2pfb_slopes.tcl',
                     'ascii2pfb_Ks.tcl',
                     'ascii2pfb_poros.tcl']
    # standard directories when no DA would be used
    # standard would be in the input_pf or namelists folder
    # with DA this would be dir_real (realization folder)
    dirs_var = [os.path.join(dir_setup,'input_pf'), #replaced namelists by input_pf
                os.path.join(dir_setup,'input_pf'),
                os.path.join(dir_setup,'input_pf'),
                os.path.join(dir_setup,'input_pf'),
                os.path.join(dir_setup,'input_pf')]
        
    Ks_tensor_st = 1000.
        
    for file_,dir_var in zip(files_parflow,dirs_var):

        #namelist: is Ks_tensor is optimized, take the namelist from the realization folder
        if 'coup_oas' in file_:
            if 'Ks_tensor' in settings_pfl['param_names']:
                dir_var = dir_real        
        
        file_path = os.path.join(dir_var,file_)
        with open( file_path , 'r') as file :
          filedata = file.read()

        ### Change some settings in case parameters are to be assimilated
        # namelist
        if 'coup_oas' in file_:
            if 'Ks_tensor' in settings_pfl['param_names']:
                pass
            else:
                filedata = filedata.replace('__Ks_tensor__', '%.1f'%Ks_tensor_st) #Ks_tensor not assimilated -> standard value
        
        # if parameter is assimilated -> set parameter directory to the realization directory
        if 'slopes' in file_:
            if 'slope_anom' in settings_pfl['param_names']:
                dir_var = dir_real
        if 'Ks' in file_:
            if 'Ks_anom' in settings_pfl['param_names'] or 'Ks' in settings_pfl['param_names']:
                dir_var = dir_real
        if 'poros' in file_:
            if 'poros_anom' in settings_pfl['param_names']:
                dir_var = dir_real
        
        # print('In file %s, %s' % file_, file_path)
        # print('Replace __dir_var__ by %s' % dir_var)
        
        filedata = filedata.replace('__nprocx_pfl_bldsva__', '%i'%settings_pfl['n_proc_pfl_x'])
        filedata = filedata.replace('__nprocy_pfl_bldsva__', '%i'%settings_pfl['n_proc_pfl_y'])
        filedata = filedata.replace('__pfl_dx__', settings_pfl['dx'])
        filedata = filedata.replace('__pfl_dy__', settings_pfl['dy'])
        filedata = filedata.replace('__pfl_dz__', settings_pfl['dz'])
        filedata = filedata.replace('__pfl_nx__', '%i'%settings_pfl['nx'])
        filedata = filedata.replace('__pfl_ny__', '%i'%settings_pfl['ny'])
        filedata = filedata.replace('__pfl_nz__', '%i'%settings_pfl['nz'])

        filedata = filedata.replace('__dir_bin__', '%s'%dir_bin)
        
        filedata = filedata.replace('__dir_pfl__', '%s'%os.path.join(dir_build,'parflow') )
        filedata = filedata.replace('__dir_pfin__', '%s'% os.path.join(dir_setup,'input_pf') )
        filedata = filedata.replace('__dir_run__', '%s'% dir_run )
        filedata = filedata.replace('__dir_var__', '%s'%dir_var)
        
        filedata = filedata.replace('__time_totalrun__', '%i'% (settings_pfl['t_total'].total_seconds()/(60*60)) )
        filedata = filedata.replace('__time_pfl_dump__', '%.1f'% (settings_pfl['t_dump'].total_seconds()/(60*60)) )
        filedata = filedata.replace('__time_couple__', '%.2f'% (settings_pfl['t_couple'].total_seconds()/(60*60)) )   
        filedata = filedata.replace('__filename_pfl_out__', settings_pfl['filename_pfl_out'] )

        filedata = filedata.replace('__icpres_type__', '%s'%settings_pfl['icpres_type'])
        filedata = filedata.replace('__geom_icpres_valorfile__', '%s'%settings_pfl['geom_icpres_valorfile'])
        filedata = filedata.replace('__geom_icpres_val__', '%s'%settings_pfl['geom_icpres_val'])
        
        
        with open(os.path.join(dir_run,file_), 'w') as file:
          file.write(filedata)
        os.chmod(os.path.join(dir_run,file_),0o755)

def make_parflow_executable(dir_run,settings_run):
    str_cmd = '''
    source %s
    tclsh %s
    tclsh %s
    tclsh %s
    tclsh %s
    tclsh %s
    ''' % (settings_run['env_file'],
           os.path.join(dir_run,'ascii2pfb_slopes.tcl'),
           os.path.join(dir_run,'ascii2pfb_SoilInd.tcl'),
           os.path.join(dir_run,'coup_oas.tcl'),
           os.path.join(dir_run,'ascii2pfb_Ks.tcl'),
           os.path.join(dir_run,'ascii2pfb_poros.tcl'))

    os.system(str_cmd)    

def adjust_oasis_files(dir_run,settings_run,settings_clm,settings_pfl):
    
    dir_setup = settings_run['dir_setup']
    
    with open(os.path.join(dir_setup,'input_oas/namcouple_pfl_clm'), 'r') as file : #replaced namelists by input_oas
      filedata = file.read()

    filedata = filedata.replace('__runTime__', '%i'% settings_clm['t_total'].total_seconds())
    filedata = filedata.replace('__cplfreq__', '%i'% settings_clm['t_couple'].total_seconds())
    filedata = filedata.replace('__ngpflx__', '%i'% settings_clm['nx'])
    filedata = filedata.replace('__ngpfly__', '%i'% settings_clm['ny'])
   
    if 'eCLM' in settings_run['models']:
        filedata = filedata.replace('__ngclmx__', '%i'% settings_clm['nx'])
        filedata = filedata.replace('__ngclmy__', '%i'% settings_clm['ny'])
    else:
        filedata = filedata.replace('__ngclmx__', '%i'% (settings_clm['nx']*settings_clm['ny'])) #the clm3.5 grid is 'flattened' in oasis
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
    
def adjust_run_files(settings_run,dir_run,settings_clm,settings_pfl,settings_sbatch):

    dir_setup = settings_run['dir_setup']
    
    if settings_run['models'] == 'eCLM': #eCLM only! 
        n_proc_clm = settings_clm['n_proc_clm']
        n_nodes = int(np.ceil( (n_proc_clm) / 48 )) # 48 cores per node on JUWEL
    
        with open( os.path.join(dir_setup,'input_slurm/tsmp_slm_run.bsh') , 'r') as file :
          filedata = file.read()

        filedata = filedata.replace('__ntasks__', '%i'%(n_proc_clm))
        filedata = filedata.replace('__nnodes__', '%i'%n_nodes)
        filedata = filedata.replace('__stime__', settings_sbatch['sbatch_time'])
        filedata = filedata.replace('__spart__', settings_sbatch['sbatch_partition'])
        filedata = filedata.replace('__sacc__', settings_sbatch['sbatch_account'])

        with open(os.path.join(dir_run,'tsmp_slm_run.bsh'), 'w') as file:
          file.write(filedata)
        os.chmod(os.path.join(dir_run,'tsmp_slm_run.bsh'),0o755)

    else: #CLM3.5-PFL or eCLM-PFL
        n_proc_pfl = settings_pfl['n_proc_pfl_x']*settings_pfl['n_proc_pfl_y']*settings_pfl['n_proc_pfl_z']
        n_proc_clm = settings_clm['n_proc_clm']
        proc_pfl_0 = 0
        proc_pfl_n = n_proc_pfl - 1 
        proc_clm_0 = n_proc_pfl
        proc_clm_n = n_proc_pfl + n_proc_clm - 1

        with open( os.path.join(dir_setup,'input_slurm/slm_multiprog_mapping.conf') , 'r') as file : #replaced namelists by input_slurm
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

        with open( os.path.join(dir_setup,'input_slurm/tsmp_slm_run.bsh') , 'r') as file :
          filedata = file.read()

        filedata = filedata.replace('__ntasks__', '%i'%(n_proc_pfl + n_proc_clm))
        filedata = filedata.replace('__nnodes__', '%i'%n_nodes)
        filedata = filedata.replace('__stime__', settings_sbatch['sbatch_time'])
        filedata = filedata.replace('__spart__', settings_sbatch['sbatch_partition'])
        filedata = filedata.replace('__sacc__', settings_sbatch['sbatch_account'])

        with open(os.path.join(dir_run,'tsmp_slm_run.bsh'), 'w') as file:
          file.write(filedata)
        os.chmod(os.path.join(dir_run,'tsmp_slm_run.bsh'),0o755)
    
def start_run(settings_run):
    sbatch_file = 'tsmp_slm_run.bsh'
    dir_build = settings_run['dir_build']
    
    str_cmd = '''
    source %s
    sbatch %s
    ''' % (settings_run['env_file'],
           sbatch_file)
    os.system(str_cmd)    
    
def wait_for_run(dir_run,settings_sbatch):
    while not os.path.exists(os.path.join(dir_run,'ready.txt')):
        print('Still running (%s)...'%(dir_run.split(os.path.sep)[-4:-1]),flush=True)
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
    
def remove_misc_files(dir_run,settings_run):
    # [os.remove(file) for file in glob(os.path.join(dir_run,'CLMSAT*.nc'))]
    # Keep all output files
    retain = ['mpiMPMD*','cordex*.out.0*','*.clm2.h[0-2].*.nc','coup_oas*','lnd*','ready*',
             'timing','log','*slm*']
    # remove = ['*.clm2.h0.2019-0[1-5]*.nc']
    # remove = ['bladiebla']
    remove = settings_run['files_remove']
    
    files_retain = []
    for retain_ in retain:
        files_retain.extend(glob(os.path.join(dir_run,retain_)))

    files_remove = []
    for remove_ in remove:
        files_remove.extend(glob(os.path.join(dir_run,remove_)))

    #Keep the last CLM restart file
    if len(glob(os.path.join(dir_run,'*.clm2.r.*.nc')))>0:
        files_retain.append(sorted(glob(os.path.join(dir_run,'*.clm2.r.*.nc')))[-1])
    if len(glob(os.path.join(dir_run,'*.clm2.rh0.*.nc')))>0:
        files_retain.append(sorted(glob(os.path.join(dir_run,'*.clm2.rh0.*.nc')))[-1])

    # files_retain.append(sorted(glob(os.path.join(dir_run,'clm.clm2.r*.nc'))) )
    
    # Check for all files if necessary to retain, if not delete
    files = os.listdir(dir_run)
    for file_ in files:
        if os.path.isfile(os.path.join(dir_run,file_)):

            if os.path.join(dir_run,file_) not in files_retain:
                # print(os.path.join(dir_run,file_))
                os.remove(os.path.join(dir_run,file_))
            elif os.path.join(dir_run,file_) in files_remove: #files that are explicitly removed, even if in the retain list
                # print(os.path.join(dir_run,file_))
                os.remove(os.path.join(dir_run,file_))              
    
def setup_submit_wait(i_real,settings_run,settings_clm,settings_pfl,settings_sbatch,date_results_iter):
    # print('Sleeping %i seconds' % i_real)
    # time.sleep(i_real)
    
    # in case of open loop mode, set the parameters to change to an empty array (in this case the standard values are used in adjust_parflow_files)
    # Make sure all required .sa files are there in the input_pf folder
    if settings_run['mode'] == 'OL':
        settings_pfl['param_names'] = []
        settings_clm['param_names'] = []
    
    dir_real = os.path.join(settings_run['dir_iter'],'R%3.3i'%i_real)
    
    #try:
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
            if settings_clm['IC_file']:
                settings_clm['file_restart'] = '%s' % settings_clm['IC_file'] 
            else:
                settings_clm['file_restart'] = ''

            if settings_pfl['IC_file']:
                settings_pfl['icpres_type'] = 'NCFile'
                settings_pfl['geom_icpres_valorfile'] = 'FileName'
                settings_pfl['geom_icpres_val'] = '"%s"' % settings_pfl['IC_file']
            else: #fall back to cold start
                settings_pfl['icpres_type'] = 'HydroStaticPatch'
                settings_pfl['geom_icpres_valorfile'] = 'Value'
                settings_pfl['geom_icpres_val'] = '-2.0'

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

            files_clm_restart = sorted(glob(os.path.join(dir_run_prev,'*.clm2.r.*.nc') ))
            print(dir_run_prev)
            print(files_clm_restart)
            print(files_clm_restart[-1].split('.'))
            date_restart = pd.to_datetime(os.path.basename(files_clm_restart[-1]).split('.')[-2][0:10]).date()
            assert date_restart == date_end_prev.date(), 'Restart date does not correspond to the expected date, %s, %s' %(date_restart,date_end_prev.date() )
            print('Restarting CLM from %s' % files_clm_restart[-1])
            settings_clm['file_restart'] = files_clm_restart[-1]
            settings_clm['init_interp'] == False # not necessary

            if 'PFL' in settings_run['models']:
                settings_pfl['icpres_type'] = 'NCFile'
                settings_pfl['geom_icpres_valorfile'] = 'FileName'    
                settings_pfl['geom_icpres_val'] = sorted(glob(os.path.join(dir_run_prev,
                                                                           'cordex%ix%i_%s.out.0*' % (settings_pfl['nx'],settings_pfl['ny'],str_date_prev))))[-1]


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

            copy_binaries(settings_run,dir_run)

            if 'eCLM' in settings_run['models']:
                settings_clm['t_total'] = date_end_sim-date_start_sim
                settings_clm['datetime_start'] = date_start_sim
                settings_clm['datetime_end'] = date_end_sim
                adjust_eclm_files(settings_run,dir_run,dir_real,settings_clm)
            else: #CLM3.5-PFL
                settings_clm['t_total'] = date_end_sim-date_start_sim
                settings_clm['datetime_start'] = date_start_sim
                adjust_clm_files(settings_run['dir_setup'],dir_run,settings_clm)
                copy_oasis_files(settings_run['dir_setup'],dir_run)

            if 'PFL' in settings_run['models']:
                settings_pfl['t_total'] = date_end_sim-date_start_sim
                settings_pfl['filename_pfl_out'] = 'cordex%ix%i_%s' % (settings_pfl['nx'],settings_pfl['ny'],str_date)
                adjust_parflow_files(settings_run['dir_setup'],dir_run,
                                     settings_run['dir_build'],settings_run['dir_binaries'],
                                     dir_real,settings_pfl)
                make_parflow_executable(dir_run,settings_run)

                adjust_oasis_files(dir_run,settings_run,settings_clm,settings_pfl)


            adjust_run_files(settings_run,dir_run,settings_clm,settings_pfl,settings_sbatch)

            os.chdir(dir_run)
            start_run(settings_run)
            ## in tsmp_slm_run.bsh, a file ready.txt is written at the end of the run: wait for this file
            wait_for_run(dir_run,settings_sbatch) 
            os.chdir(settings_run['dir_setup'])

            if settings_run['ndays_spinup'] is not None and i1 == 0:
                # i1 is the spinup -> results not useful, remove
                settings_run_ = copy.deepcopy(settings_run) #make deep copy: otherwise all files will be removed next iterations 
                settings_run_['files_remove'].append('*.clm2.h[0-2].*.nc')
            else:
                settings_run_ = copy.deepcopy(settings_run)
            remove_misc_files(dir_run,settings_run_)

            # ## Last step: move the run directory to storage (scratch), and keep a link
            # if not os.path.exists(settings_run['dir_store']):
            #     print('Creating dir: %s' % settings_run['dir_store'])
            #     os.makedirs(settings_run['dir_store'])
            # print('Moving files to storage: %s' % settings_run['dir_store'])
            # move_and_link(dir_run,settings_run['dir_store'])


        else:
            print('%s exists, continuing...' % dir_run) 
   
    #except Exception as e:
    #    print(f"An exception occurred: {e}")
                
            
if __name__ == '__main__':
    
    settings_run['dir_real'] = '/p/scratch/cjibg36/kaandorp2/TSMP_results/TSMP_patched/DA_tsmp_cordex_111x108/20190102-20191231/i000/R000'
    date_results_iter = date_results_binned[0]
    
    setup_submit_wait(settings_run,settings_clm,settings_pfl,settings_sbatch,date_results_iter)
