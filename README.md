# TSMP_DA

![schematic structure TSMP_DA](https://github.com/mikaelk/TSMP_DA/blob/master/TSMP_DA.png?raw=true)

**Main steps to get the data assimilation working**
First install eCLM, and copy important files:
1. Install eCLM with parameter changes: https://github.com/mikaelk/eCLM, up to and including the cmake steps (namelist generator not required), into your $eCLM_dir. In order to build eCLM, load the environment file from https://github.com/HPSCTerrSys/TSMP2/tree/master/env for the right packages
2. Copy the forcing data from /p/largedata/jibg36/kaandorp2/setup_TSMP_DA/ERA5_EUR-11_CLM/ to your $forcing_dir
3. Copy the eCLM setup template folder from /p/largedata/jibg36/kaandorp2/setup_TSMP_DA/setup_eclm_cordex_444x432_v8 to your $template_dir
4. Copy the misc files from /p/largedata/jibg36/kaandorp2/setup_TSMP_DA/ somewhere (env file & restart file)

Some of the most important run settings to adjust:
1. In DA/settings.py adjust settings_run['dir_build'] to your $eCLM_dir
2. In settings_run['dir_binaries'] also adjust where the binary files are located (somewhere within $eCLM_dir)
3. In settings_run['env_file'] specify the environment file that needs to be loaded to run eCLM (copy from /p/largedata/jibg36/kaandorp2/setup_TSMP_DA/)
4. In DA/settings.py adjust settings_run['dir_forcing'] to your $forcing_dir
5. In DA/settings.py adjust settings_run['dir_template'] to your $template_dir
6. In DA/settings.py specify where your run should be created in settings_run['dir_setup']
7. If settings_run['init_restart'] is True, make sure to specify IC_file_CLM correctly in settings.py
8. We need some common clm files (e.g. related to population density, methane). Copy these from /p/largedata/jibg36/kaandorp2/setup_TSMP_DA/, and adjust the settings_clm['dir_common_eclm'] setting

Some additional files are required for the data assimilation operators, copy and adjust these in settings_DA:
1. Observations SMAP (settings_DA['folder_SMAP']), copy from /p/largedata/jibg36/kaandorp2/setup_TSMP_DA/SMAP
2. Observations ICOS (settings_DA['folder_FLX']), copy from /p/largedata/jibg36/kaandorp2/setup_TSMP_DA/ICOS
3. Land/sea mask (settings_DA['file_lsm']), copy from /p/largedata/jibg36/kaandorp2/setup_TSMP_DA/static/
4. File with grid corner points (settings_DA['file_corner']) , copy from /p/largedata/jibg36/kaandorp2/setup_TSMP_DA/static/

**To do together**:
1. Show jupyter notebook, easy debugging and checking output figures

2. Do a simple run by adjusting the above settings
    + adjust settings
    + start a screen in bash: screen -S session_name
    + remember the login node in which screen is running if you want to go back to it later
    + cd to DA folder, source modules.sh
    + python -u main_DA.py
    + ctrl-a-d to exit screen
    + screen -ls: view screen
    + screen -r session_name
    + If a run crashes, remove the folder (that you specified in settings_run['dir_setup']) and try again
    
3. Include a new parameter from the clm5_params.c171117.nc file in the assimilation
    + When adding a new parameter, we can make a new template folder (in case something breaks, of when simulations are still running, nothing goes wrong)
    + I also like to have a separate repository for eCLM when making changes
    + modify the parameter file (clm5_params.c17117.nc) 
    + Include the new parameter in setup_parameters.py
    + Include the parameter in generate_parameters.py
    + Include the parameter in the header of settings.py
    + Include the parameter in run_realization.py
    + Don't forget to change dir binaries etc. in settings.py if you made a new repo for eCLM, and adjust the template folder if you made a new one
    
    
**Crashed runs**
When a run has crashed (e.g. JUWELS broke down) the easiest thing to do:
1. Either delete the folder with the last iteration that stopped prematurely (e.g. 'i003')
2. Or delete the run folders for which no restart file is available (e.g. check by using wildcards: 'ls 20190101-20191230/i003/**/run_20190101-20191230/EU11.clm2.r.2019-12-30-00000.nc' -> check which folders do not contain the restart files -> delete those run folders)
3. Adjust the settings.py file. Open the copy of the settings file that was made in the folder 'settings'. *Important*: if the inflation factor (settings_DA['alpha']) was already calculated (i.e. is not 'None' in the settings file), make sure this is copied! 
4. Go back to your screen, and simply run 'python -u main_DA.py' again


**Extra**
when adjusting eCLM source code, you can rebuild the code by using
1. cd ${BUILD_DIR}
2. make -j8 install


**Future steps**
Some future steps that could be implemented:
1. State vector DA. There are two options:
    + Use the ensemble of the last iteration to do a offline-DA of state vectors of interest. E.g. simply let the Kalman filter take a combination of the soil moisture as predicted by the different ensemble members over a year that matches SMAP the best. The good thing is that the final ensemble is relatively bias-free, and should be centered around the observations in most locations
    + 'online' DA: run the model for e.g. 1 week. Update the state vector, and write it back to the CLM restart file. Should be doable, but the format of the restart files needs to be looked at. The 'date loop' in main_DA.py needs to be worked out (i_date is now fixed to 0)
2. Run eCLM in BGC mode, assimilate LAI by comparing to e.g. MODIS. A new operator would need to be implemented mapping LAI predicted by CLM to the MODIS observation locations


**Single site runs**
Currently the setup works for EUROCORDEX, with 1 simulation running on 1 node of JUWELS (using 'srun eclm.exe' called in 'tsmp_slm_run.bsh'). Single site setups are in theory possible, but would each be run on one node at the moment (wasting large amount of resources), so this would need to be changed. Steps to take:
1. in main_DA.py, take a look where 'setup_submit_wait' is called. This function is currently called in parallel. In the case of single site runs, I would recommend running as many simulations as possible on a single node. So in that case, this function would perhaps only be called 1 single time, instead of parallel
2. in run_realization.py -> setup_submit_wait, implement a new version that is not run for each ensemble member (realization) separately. That means, that 
    + for each ensemble member, create dir_run, call the functions copy_binaries, adjust_eclm_files like normally
    + adjust_run_files needs to be changed. For example, for each ensemble member, append a line in 'tsmp_slm_run.bsh' (e.g. using job steps https://apps.fz-juelich.de/jsc/hps/juwels/batchsystem.html#job-steps). The location of 'tsmp_slm_run.bsh' should be adjusted, perhaps it should be place within the iteration directory (e.g. 'DA_folder/20190101-20200101/i000' for iteration 0).
    + start_run can then be called on the adjusted 'tsmp_slm_run.bsh' file
    + wait_for_run needs to check if 'ready.txt' is available in all run folders for all ensemble members (e.g. within 'DA_folder/20190101-20200101/i000/R000...R064/run_folder' )
3. The DA operators need to be changed, or written from scratch
4. This will be a bit complex, so I would recommend doing all of these steps line for line in a jupyter notebook, otherwise it will be impossible to debug