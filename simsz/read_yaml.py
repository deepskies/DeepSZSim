import yaml
from typing import Union
import os

class YAMLOperator(object):

    def __init__(self, file_path= Union[None, str]):

        self.file_path = file_path

    def parse_yaml(self):
        """
        Parse a YAML file and return a dictionary.

        Args:
            file_path (str): Path to the YAML file.

        Returns:
            dict: Dictionary containing the parsed YAML file.

        Raises:
            yaml.YAMLError: If the YAML file is not valid.
        """
        with open(self.file_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def load_vars(self, ref):
        survey=[key for key in ref['SURVEYS'].keys()][0]
        survey_freq=[key for key in ref['SURVEYS'][survey].keys()][0]
        beam_size=ref['SURVEYS'][survey][survey_freq]['beam_size']
        noise_level=ref['SURVEYS'][survey][survey_freq]['noise_level']
        image_size = ref['IMAGES']['image_size']
        pixel_scale = ref['IMAGES']['pixel_scale']
        for key in ref['COSMOLOGY'].keys():
            cosmo_dict=ref['COSMOLOGY'][key] #Read in cosmological parameters

        sigma8=cosmo_dict['sigma8']
        ns=cosmo_dict['ns']

        cosmo=FlatLambdaCDM(cosmo_dict['H0'], cosmo_dict['Omega_m0'], Tcmb0=cosmo_dict['t_cmb'], Ob0=cosmo_dict['Omega_b0'])
        return(survey,survey_freq,beam_size,noise_level,image_size,pixel_scale,cosmo,sigma8,ns)
