"""
unit conversions and data saving functions
"""

import numpy as np
import os

def Mpc_to_arcmin(r_arcmin, redshift_z, cosmo):
    '''
    Changes the units of r from Mpc to arcmin

    Parameters:
    -----------
    r_arcmin: float or array
        the distance r, in units of arcmin
    redshift_z: float
        the redshift (unitless)
    cosmo: FlatLambaCDM instance
        background cosmology for density calculation

    Returns:
    --------
    r_arcmin / Mpc_per_arcmin: float or array (same type as r_arcmin)
        r_arcmin in units of Mpc
    '''
    Kpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(redshift_z).value
    Mpc_per_arcmin = Kpc_per_arcmin/1000.

    return r_arcmin / Mpc_per_arcmin


def arcmin_to_Mpc(r_Mpc, redshift_z, cosmo):
    '''
    Changes the units of r from arcmin to Mpc

    Parameters:
    -----------
    r_Mpc: float or array
        the distance r, in units of Mpc
    redshift_z: float
        the redshift (unitless)
    cosmo: FlatLambaCDM instance
        background cosmology for density calculation

    Returns:
    --------
    r_Mpc / Mpc_per_arcmin: float or array (same type as r_Mpc)
        r_Mpc in units of arcmin
    '''
    Kpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(redshift_z).value
    arcmin_per_Mpc = 1000/Kpc_per_arcmin
    return r_Mpc / arcmin_per_Mpc


def gaussian_kernal(pix_size_arcmin,beam_size_fwhp_arcmin):
    '''
    Parameters:
    -----------
    pix_size_arcmin: float
        size of each pixel in arcmin
    beam_size_fwhp_arcmin: float
        beam size in arcmin

    Returns:
    --------
    gaussian: array
        2d gaussian kernal
    '''
    N=37
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) * pix_size_arcmin
    X = np.outer(ones, inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.)

    beam_sigma = beam_size_fwhp_arcmin / np.sqrt(8.*np.log(2))
    gaussian = np.exp(-.5 *(R/beam_sigma)**2.) / (2 * np.pi * (beam_sigma ** 2))
    gaussian = gaussian / np.sum(gaussian)

    return(gaussian)

def save_sim_to_h5(file, name, data, attributes={}, overwrite=False):
    '''
    Save simulated data to h5 file

    Parameters:
    -----------
    file : h5py.File
        The HDF5 file where the data will be saved.
    name : str
        The name under which the data will be stored in the HDF5 file.
    data : dict
        A dictionary where keys are dataset names (strings) and values are 
        the corresponding dataset arrays or values.

    Returns:
    --------
    None   
    '''

    if overwrite and name in file:
        del file[name]
    for dname, dset in data.items():
        file.create_dataset(os.path.join(name, dname), data=dset)
    if name not in file:
        file.create_group(name)
    for aname, attr in attributes.items():
        file[name].attrs[aname] = attr
