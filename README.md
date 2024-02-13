# TSMP_DA

![schematic structure TSMP_DA](https://github.com/mikaelk/TSMP_DA/blob/master/TSMP_DA.png?raw=true)

**Main steps to get the data assimilation working**
First install eCLM, and copy important files:
1. Install eCLM with parameter changes: https://github.com/mikaelk/eCLM, up to and including the cmake steps (namelist generator not required), into your $eCLM_dir
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
    + cd to DA folder, source modules.sh
    + python -u main_DA.py
    + ctrl-a-d to exit screen
    + screen -ls: view screen
    + screen -r session_name
    
3. Include a new parameter from the clm5_params.c171117.nc file in the assimilation
    + Include the new parameter in setup_parameters.py
    + Include the parameter in generate_parameters.py
    + Include the parameter in the header of settings.py
    + Include the parameter in run_realization.py
    
**Extra**
when adjusting eCLM source code, you can rebuild the code by using
1. cd ${BUILD_DIR}
2. make -j8 install