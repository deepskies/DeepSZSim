"""
beam convolution, SZ function and CMB simulation tools 
"""

from deepszsim import utils
import scipy
import numpy as np
import astropy.constants as c
from astropy import units as u
from pixell import enmap
import camb

def convolve_map_with_gaussian_beam(pix_size_arcmin,
                                    beam_size_fwhm_arcmin, map_to_convolve):
    '''
    Parameters:
    ----------
    pix_size_arcmin: float
        size of each pixel in arcmin
    beam_size_fwhm_arcmin: float
        beam size in arcmin
    map_to_convolve: array
        image to apply beam convolution to

    Return:
    -------
    convolved_map: array

    Note - pixel size and beam_size need to be in the same units
    '''
    gaussian = utils.gaussian_kernal(pix_size_arcmin, beam_size_fwhm_arcmin)
    convolved_map = scipy.signal.fftconvolve(map_to_convolve, 
                                             gaussian, mode = 'same')

    return(convolved_map)

def f_sz(freq_ghz, T_CMB_K):
    '''
    the leading order correction to blackbody from Compton scattering
    see Eq 3.31 of https://background.uchicago.edu/~whu/thesis/chap3.pdf
    
    Parameters:
    ----------
    freq_ghz: float
        Observation frequency f, in units of GHz
    T_CMB_K: Quantity instance
        Temperature of CMB in K

    Return:
    ------
    fsz: float
        radiation frequency
    '''

    f=freq_ghz*u.GHz #Takes input in units of GHz
    f=f.to(1/u.s) #Unit conversion
    x = (c.h * f / c.k_B / T_CMB_K).value
    fsz = x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4

    return fsz

def add_cmb_map_and_convolve(dT_map_uK, ps, pix_size_arcmin, 
                                 beam_size_fwhp_arcmin):
    '''
    Parameters:
    ----------
    dT_map_uK: array
        the map to add to the CMB, units of -uK
    ps: array
        power spectrum with shape (3, 3, lmax); clTT spectrum at ps[0][0]
    pix_size_arcmin: float
        size of each pixel in arcmin
    beam_size_fwhp_arcmin: float or None
        beam size in arcmin

    Return:
    ------
    dT submap: array
        dT submap with same shape as dT_map, in units of -uK
    '''
    padding_value = int(np.ceil(beam_size_fwhp_arcmin/pix_size_arcmin))
    expanded_shape = (dT_map_uK.shape[0] + 2*padding_value, 
                        dT_map_uK.shape[1]+2*padding_value)
    #print(expanded_shape)
    cmb_map = make_cmb_map(shape=expanded_shape, 
                                pix_size_arcmin=pix_size_arcmin, ps=ps)
    if type(dT_map_uK) is u.Quantity:
        cmb_map = cmb_map *u.uK
    dT_map_expanded = np.pad(dT_map_uK, (padding_value,padding_value),  
                                constant_values=0)
    signal_map = dT_map_expanded + cmb_map

    if beam_size_fwhp_arcmin is not None:
        conv_map = convolve_map_with_gaussian_beam(
            pix_size_arcmin, beam_size_fwhp_arcmin, signal_map)
    else:
        conv_map = signal_map

    return (conv_map[padding_value:-padding_value, 
                    padding_value:-padding_value],
            cmb_map[padding_value:-padding_value, 
                    padding_value:-padding_value]
    ) 

def get_cls(ns, cosmo, lmax=2000):
    '''
    Makes a cmb temperature map based on the given power spectrum

    Parameters:
    ----------
    ns: float
        scalar spectral index of the power-spectrum
    cosmo: FlatLambaCDM instance
        background cosmology

    Return:
    ------
    ps array
        power spectrum as can be used in szluster.make_cmb_map
    '''
    data = camb.set_params(ns=ns, H0=cosmo.H0.value, ombh2=cosmo.Ob0, 
                            omch2=cosmo.Om0, lmax=lmax, WantTransfer=True)
    results = camb.get_results(data)
    cls = np.swapaxes(results.get_total_cls(CMB_unit='muK', raw_cl=True),
                        0,1)
    ps = np.zeros((3,3,cls.shape[1]))
    # Needs to be reshaped to match input for pixell.enmap
    ps[0][0]= cls[0] # clTT spectrum
    ps[1][0] = cls[3] #clTE
    ps[0][1] = cls[3] #clTE
    ps[1][1] = cls[1] #clEE
    ps[2][2] = cls[2] #clBB
    return ps

def make_cmb_map(shape, pix_size_arcmin, ps, seed=None):
    '''
    Makes a cmb temperature map based on the given power spectrum

    Parameters:
    ----------
    shape: tuple
        shape of submap in arcmin
    pix_size_arcmin: float
        size of each pixel in arcmin
    ps: array
        power spectrum with shape (3, 3, lmax); clTT spectrum at ps[0][0]

    Return:
    -------
    cmb T map: array
    '''
    #ps[0][0] is cltt spectrum
    shape,wcs = enmap.geometry(shape=shape,pos=(0,0),
                                res=np.deg2rad(pix_size_arcmin/60.))
    shape = (3,) + shape
    omap = enmap.rand_map(shape,wcs,cov=ps, seed=seed)
    #omap gives TQU maps, so for temperature, we need omap[0]

    return omap[0]

