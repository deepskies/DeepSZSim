import astropy
from astropy.cosmology import FlatLambdaCDM
import simsz.read_yaml as read_yaml
from typing import Union
import os
import sys

class load_vars:

    def __init__(self, 
                 file_path = os.path.join(os.path.dirname(__file__), "Settings", "inputdata.yaml"),
                 survey_num : int = None,
                 survey_name : str = None,
                 survey_freq_val : int = None,
                 cosmo_name : str = None):
        self.data = []
        self.file_path = file_path
        self.dict = {}
        ref = read_yaml.YAMLOperator(self.file_path).parse_yaml()
        if len(list(ref['SURVEYS'].keys())) == 1:
            survey = list(ref['SURVEYS'].keys())[0]
        elif survey_num is not None:
            survey = list(ref['SURVEYS'].keys())[survey_num]
        elif survey_name is not None:
            survey = survey_name
        else:
            print("specify a survey of interest with `survey_name` or `survey_num`")
            sys.exit()
        if len(list(ref['SURVEYS'][survey].keys())) == 1:
            survey_freq = list(ref['SURVEYS'][survey].keys())[0]
        elif survey_freq_val is not None:
            survey_freq = survey_freq_val
        else:
            print("specify a survey frequency of interest with `survey_freq_val`")
            sys.exit()
        self.dict["survey"] = survey
        self.dict["survey_freq"] = survey_freq
        self.dict["beam_size_arcmin"] = ref['SURVEYS'][survey][survey_freq]['beam_size']
        self.dict["noise_level"] = ref['SURVEYS'][survey][survey_freq]['noise_level']
        self.dict["image_size_arcmin"] = ref['IMAGES']['image_size']
        self.dict["pix_size_arcmin"] = ref['IMAGES']['pix_size']
        if len(list(ref['COSMOLOGY'].keys())) == 1:
            cosmo_dict = ref['COSMOLOGY'][list(ref['COSMOLOGY'].keys())[0]] #Read in cosmological parameters
        elif cosmo_name is not None:
            cosmo_dict = ref['COSMOLOGY'][cosmo_name]
        else:
            print("specify cosmology name with `cosmo_name`")
            sys.exit()
        if cosmo_dict['flat']:
            self.dict["cosmo"] = FlatLambdaCDM(cosmo_dict['H0'], cosmo_dict['Omega_m0'], Tcmb0 = cosmo_dict['t_cmb'],
                                               Ob0 = cosmo_dict['Omega_b0'])
        else:
            print("only flat cosmology supported at this time")
            self.dict["cosmo"] = FlatLambdaCDM(cosmo_dict['H0'], cosmo_dict['Omega_m0'], Tcmb0 = cosmo_dict['t_cmb'],
                                               Ob0 = cosmo_dict['Omega_b0'])
        self.dict["sigma8"] = cosmo_dict['sigma8']
        self.dict["ns"] = cosmo_dict['ns']
