"""
read-in functions for input parameter yaml and h5 files  
"""

import astropy
from astropy.cosmology import FlatLambdaCDM
import deepszsim.read_yaml as read_yaml
from typing import Union
import os
import sys
import h5py

def load_vars(file_name= "config_simACTDR5.yaml",
              file_dir = os.path.join(os.path.dirname(__file__), "Settings"),
              survey_num : int = None,
              survey_name : str = None,
              survey_freq_val : int = None,
              cosmo_name : str = None,
              enforce_odd_pix : bool = True):
    """
    
    Parameters
    ----------
    file_name: str
        name of a yaml file that specifies the variables that you want to simulate
    file_dir: str
        path to the yaml file named in `file_name`
    survey_num: None or int
        optional -- if there are multiple surveys, choose one by its position in the list
    survey_name: None or str
        optional -- if there are multiple surveys, choose one by its name
    survey_freq_val: None or int
        optional -- if a survey has multiple frequencies, choose a frequency by its value (in GHz)
    cosmo_name: None or str
        optional -- if there are multiple cosmologies, choose one by its name
    enforce_odd_pix: bool
        whether or not to enforce an odd number of pixels in the simulated image (defaults to true so that the
        central pixel has the maximal y-value

    Returns
    -------
    dict
        a dictionary that will enable you to simulate SZ clusters with a given cosmology and set of observational
        properties.

    """
    file_path = os.path.join(file_dir, file_name)
    dict = {}
    ref = read_yaml.YAMLOperator(file_path).parse_yaml()
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
    dict["survey"] = survey
    dict["survey_freq"] = survey_freq
    dict["beam_size_arcmin"] = ref['SURVEYS'][survey][survey_freq]['beam_size']
    dict["noise_level"] = ref['SURVEYS'][survey][survey_freq]['noise_level']
    img_params_specified = list(ref['IMAGES'].keys())
    img_params_all = ('image_size_pixels', 'image_size_arcmin', 'pixel_size_arcmin')
    if len(img_params_specified) == 2:
        for p in img_params_specified:
            dict[p] = ref['IMAGES'][p]
        if img_params_all[0] not in img_params_specified:
            dict[img_params_all[0]] = dict[img_params_all[1]]/dict[img_params_all[2]]+1
        elif img_params_all[1] not in img_params_specified:
            dict[img_params_all[1]] = (dict[img_params_all[0]]-1)*dict[img_params_all[2]]
        else:
            dict[img_params_all[2]] = dict[img_params_all[1]]/(dict[img_params_all[0]]-1)
    elif len(img_params_specified) < 2:
        print(f"two of {img_params_all} should be specified -- exiting")
        return None
    else:
        print(f"only two of {img_params_all} should be specified -- using `image_size_pixels` and `pixel_size_arcmin`")
        for p in img_params_specified:
            dict[p] = ref['IMAGES'][p]
        dict[img_params_all[1]] = (dict[img_params_all[0]]-1) * dict[img_params_all[2]]
    if (dict[img_params_all[0]]%2 == 0) and (enforce_odd_pix):
        dict[img_params_all[0]] += 1
        dict[img_params_all[1]] = (dict[img_params_all[0]]-1)*dict[img_params_all[2]]
        print(f"you specified `image_size_pixels = {dict[img_params_all[0]] - 1}`, but we strongly encourage having an "
              f"odd number of pixels because the central value should be the maximum, so we added one to your number "
              f"of pixels, such that now `image_size_pixels = {dict[img_params_all[0]]}`. If however you are "
              f"absolutely certain you want an even number of pixels, please rerun this command with "
              f"`enforce_odd_pix` set to `False`.")
    if len(list(ref['COSMOLOGY'].keys())) == 1:
        cosmo_dict = ref['COSMOLOGY'][list(ref['COSMOLOGY'].keys())[0]] #Read in cosmological parameters
    elif cosmo_name is not None:
        cosmo_dict = ref['COSMOLOGY'][cosmo_name]
    else:
        print("specify cosmology name with `cosmo_name`")
        sys.exit()
    if not cosmo_dict['flat']:
        print("only flat cosmology supported at this time")
    dict["cosmo"] = FlatLambdaCDM(cosmo_dict['H0'], cosmo_dict['Omega_m0'], Tcmb0 = cosmo_dict['t_cmb'],
                                  Ob0 = cosmo_dict['Omega_b0'])
    dict["sigma8"] = cosmo_dict['sigma8']
    dict["ns"] = cosmo_dict['ns']
    
    return dict

def readh5(fname, fdir = None):
    """
    
    Parameters
    ----------
    fname: str
        name of the file you want to read (must include the suffix, eg '.h5' or '.hd5' or '.hdf5')
    fdir: None or str
        path to the directory that contains the file you want to read
    Returns
    -------
    dict
        dictionary that represents a single cluster (if `fname` represents a single cluster) or a nested set of
        clusters. Each cluster that is returned will itself have a dictionary with 'maps' and 'params' as keys,
        containing the output maps and the parameters that were used to make them, respectively
    """
    fdir = os.path.join(os.path.dirname(__file__), "outfiles") if fdir is None else fdir
    try:
        with h5py.File(os.path.join(fdir, fname), "r") as f:
            fmdict = {'maps':{}, 'params':{}}
            for m in f['maps']:
                fmdict['maps'][m] = f['maps/'+m][()]
            for p in f['params']:
                fmdict['params'][p] = f['params/'+p][()]
    except KeyError:
        with h5py.File(os.path.join(fdir, fname), "r") as f:
            fmdict = {}
            for c in f:
                fmdict[c] = {'maps':{}, 'params':{}}
                for m in f[c+'/maps']:
                    fmdict[c]["maps"][m] = f[c+'/maps/'+m][()]
                for p in f[c+'/params']:
                    fmdict[c]["params"][p] = f[c+'/params/'+p][()]
    return fmdict
