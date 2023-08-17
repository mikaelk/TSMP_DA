import os
import shutil


from run_realization import setup_submit_wait
from settings import settings_run,settings_clm,settings_pfl,settings_sbatch, date_iter_binned


'''
 1) Copy the folder template to the setup location if the destination does not exist
'''
if not os.path.exists(settings_run['dir_setup']):
    print('Copying folder template from %s to %s' % (settings_run['dir_template'],settings_run['dir_setup']) )
    shutil.copytree(settings_run['dir_template'],settings_run['dir_setup'])
else:
    print('Continuing simulation in %s' % settings_run['dir_setup'])

dir_settings = os.path.join(settings_run['dir_setup'],'settings')
if not os.path.exists(dir_settings):
    os.mkdir(dir_settings)
shutil.copy('settings.py',dir_settings)

settings_run['dir_iter'] = settings_run['dir_setup']

'''
 2) Submit run and wait
'''
setup_submit_wait(0,settings_run,settings_clm,settings_pfl,settings_sbatch,date_iter_binned)

