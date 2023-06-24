import yaml
import numpy as np
import re
from collections.abc import Iterable
import os

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


    def write_params_yaml_new()
        """
        write updated yaml file to disk
        """
            