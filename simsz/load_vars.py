import astropy
from astropy.cosmology import FlatLambdaCDM
import simsz.read_yaml as read_yaml
from typing import Union

class load_vars(object):

    def __init__(self, file_path= Union[None, str]):
        self.file_path = file_path
        self.data = []

    def make_dict_and_flatLCDM(self):
        ref=read_yaml.YAMLOperator(self.file_path).parse_yaml()
        survey=[key for key in ref['SURVEYS'].keys()][0]
        survey_freq=[key for key in ref['SURVEYS'][survey].keys()][0]
        beam_size=ref['SURVEYS'][survey][survey_freq]['beam_size']
        noise_level=ref['SURVEYS'][survey][survey_freq]['noise_level']
        image_size = ref['IMAGES']['image_size']
        pix_size = ref['IMAGES']['pix_size']
        for key in ref['COSMOLOGY'].keys():
            cosmo_dict=ref['COSMOLOGY'][key] #Read in cosmological parameters

        sigma8=cosmo_dict['sigma8']
        ns=cosmo_dict['ns']

        cosmo=FlatLambdaCDM(cosmo_dict['H0'], cosmo_dict['Omega_m0'], Tcmb0=cosmo_dict['t_cmb'], Ob0=cosmo_dict['Omega_b0'])
        return(survey,survey_freq,beam_size,noise_level,image_size,pix_size,cosmo,sigma8,ns)
