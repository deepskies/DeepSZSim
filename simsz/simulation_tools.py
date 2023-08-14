import utils
import scipy
import numpy as np
from astropy.constants import G, sigma_T, m_e, c, h, k_B
from astropy import units as u

def convolve_map_with_gaussian_beam(pix_size_arcmin, beam_size_fwhp_arcmin, map_to_convolve):
    '''
    Input: pixel size, beam size in arcmin, image
    Return: convolved map

    Note - pixel size and beam_size need to be in the same units
    '''
    gaussian = utils.gaussian_kernal(pix_size_arcmin, beam_size_fwhp_arcmin)
    convolved_map = scipy.signal.fftconvolve(map_to_convolve, gaussian, mode = 'same')

    return(convolved_map)

def f_sz(freq, T_CMB):
    '''
    Input: Observation frequency f in GHz, Temperature of cmb T_CMB
    Return: Radiation frequency
    '''

    f=freq*u.GHz #Takes input in units of GHz
    f=f.to(1/u.s) #Unit conversion
    x = h * f / k_B / T_CMB
    fsz = x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4

    return fsz

def get_tSZ_signal(Map, radmax, fmax=np.sqrt(2)):
    """
    Parameters:
    Map
    radmax: the radius of the inner radius
    fmax: Ratio of inner to outer radius, default of sqrt(2)

    Returns: The average value within an annulus of inner radius R, outer radius sqrt(2)*R, and the tSZ signal
    """

    center = np.array(Map.shape) / 2
    x, y = np.indices(Map.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    radius_out=radmax * fmax #define outer radius
    disk_mean = Map[r < radmax].mean() #average in inner radius
    ring_mean = Map[(r >= radmax) & (r < radius_out)].mean() #average in outer radius
    tSZ = disk_mean - ring_mean

    return disk_mean, ring_mean, tSZ