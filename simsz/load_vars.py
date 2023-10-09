import astropy
from astropy.cosmology import FlatLambdaCDM
import simsz.read_yaml as read_yaml
from typing import Union
import os

class load_vars:

    def __init__(self, 
                 file_path=os.path.join(os.path.dirname(__file__), "Settings", "inputdata.yaml")):
        self.data = []
        self.file_path = file_path

    def make_dict_and_flatLCDM(self):
        d = {}
        ref=read_yaml.YAMLOperator(self.file_path).parse_yaml()
        survey = [key for key in ref['SURVEYS'].keys()][0]
        survey_freq = [key for key in ref['SURVEYS'][survey].keys()][0]
        beam_size_arcmin = ref['SURVEYS'][survey][survey_freq]['beam_size']
        noise_level = ref['SURVEYS'][survey][survey_freq]['noise_level']
        image_size_arcmin = ref['IMAGES']['image_size']
        pix_size_arcmin = ref['IMAGES']['pix_size']
        d["survey"] = survey
        d["survey_freq"] = survey_freq
        d["beam_size_arcmin"] = beam_size_arcmin
        d["noise_level"] = noise_level
        d["image_size_arcmin"] = image_size_arcmin
        d["pix_size_arcmin"] = pix_size_arcmin
        for key in ref['COSMOLOGY'].keys():
            cosmo_dict=ref['COSMOLOGY'][key] #Read in cosmological parameters

        sigma8=cosmo_dict['sigma8']
        ns=cosmo_dict['ns']

        cosmo=FlatLambdaCDM(cosmo_dict['H0'], cosmo_dict['Omega_m0'], Tcmb0=cosmo_dict['t_cmb'], Ob0=cosmo_dict['Omega_b0'])
        d["sigma8"] = sigma8
        d["ns"] = ns
        d["cosmo"] = cosmo
        return d
