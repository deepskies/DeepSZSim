"""
CAMB and yaml-handling tools  
"""

import yaml
import numpy as np
import re
from collections.abc import Iterable
import os

class config_obj:
    """
    configuration object that is used to obtain power spectra

    Attributes
    ----------
    CAMBparams : CAMBparams instance
        this is necessary for CAMB to return results
    UserParams : dict
        a dictionary of values that the user has specified (smaller than the corresponding
        dictionary that would be necessary to fully specify a CAMBparams instance)
    dict_iterables : dict
        a dictionary of all of the iterables that the user has specified, which will be made
        available to loop over in camb_power_spectrum.CAMBPowerSpectrum
    """
    def __init__(
            self,
            user_config=os.path.join(os.path.dirname(__file__), "settings", "user_config.yaml")
    ):
        """

        Parameters
        ----------
        user_config : str
            path to yaml file that contains params the user wants to change
        base_config : str
            path to yaml file that contains baseline cosmological parameters that reflect the best-fit
            2018 Planck cosmology and which instruct CAMB to calculate useful observables. A full list
            is available at https://camb.readthedocs.io/en/latest/model.html
        """
        self._all_params_dict = {
            'USERPARAMS': _quick_yaml_load(user_config),
        }

        self._outdir = self.UserParams['outfile_dir']

                
        if len(self._all_params_dict['USERPARAMS']) > 0:
            try:
                for x, y in self._all_params_dict['USERPARAMS']['ITERABLES'].items():
                    if isinstance(y, Iterable) and len(y) <= 3 and type(y[-1]) == int:
                        self._all_params_dict['USERPARAMS']['ITERABLES'][x] = np.linspace(*y)
                    else:
                        if not isinstance(y, Iterable):
                            print(x, "is not iterable; are you sure it should be in ITERABLES?")
                        self._all_params_dict['USERPARAMS']['ITERABLES'][x] = np.array(y)
            except KeyError:
                print("no iterables specified")
                self._all_params_dict['USERPARAMS']['ITERABLES'] = {}

        self.UserParams = self._all_params_dict['USERPARAMS']

        self.dict_iterables = self._all_params_dict['USERPARAMS']['ITERABLES']  # make this more easily accessible


    def update_val(self, attr, new_val):
        """
        updates values in the config_obj
        Parameters
        ----------
        attr : str
            a key of the UserParams dictionary
        new_val : float
            new value that you wish attr to take on
        """
        attr_split = re.split("\.", attr)
        if attr in self._all_params_dict['USERPARAMS']:
            self._all_params_dict['USERPARAMS'][attr] = new_val
            self.UserParams[attr] = new_val
            print(f"updated {attr} in UserParams")
        else:
            print("not a valid attribute")


    def write_params_yaml_new():
        """
        write updated yaml file to disk
        incorporate run id
        """
        with open(os.path.join(savedir, f"{run_id}_params.yaml"), permission) as yaml_file:
            dump = pyyaml.dump(self.dict, default_flow_style = False, allow_unicode = True, encoding = None)
            yaml_file.write( dump )

            
    def _generate_run_id(random_digits=6):
        '''
        '''

        _rint = np.random.randint(10**random_digits)
        runid = 'runid_'+dt.now().strftime('%y%m%d%H%M%S%f_')+str(_rint).zfill(random_digits)

        return runid


    def cosmology_param(self, ref):
        """
        """
        
        for key in ref['COSMOLOGY'].keys():
            cosmo_dict=ref['COSMOLOGY'][key] #Read in cosmological parameters
        
        sigma8=cosmo_dict['sigma8']
        ns=cosmo_dict['ns']
    
        cosmo=FlatLambdaCDM(cosmo_dict['H0'], cosmo_dict['Omega_m0'], Tcmb0=cosmo_dict['t_cmb'], Ob0=cosmo_dict['Omega_b0']) 
        
        return (cosmo,sigma8,ns)


    def _quick_yaml_load(infile):
        """
        simply load yaml files without remembering the syntax or yaml.safe_load command
        Parameters
        ----------
        infile : str
            path to yaml file that you wish to load
        Returns
        -------
        dict
            a "safe load" dictionary version of infile
        """
        with open(infile, "r") as f:
            return yaml.safe_load(f)
