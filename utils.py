import numpy as np

def Mpc_to_arcmin(r, z, cosmo):
    '''
    Input: distance r, redshift
    Return: angular scale
    '''
    Kpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).value
    Mpc_per_arcmin = Kpc_per_arcmin/1000.

    return r / Mpc_per_arcmin


def arcmin_to_Mpc(r, z,cosmo):
    '''
    Reverse of Mpc_to_arcmin
    '''
    Kpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).value
    arcmin_per_Mpc = 1000/Kpc_per_arcmin
    return r / arcmin_per_Mpc


def gaussian_kernal(pix_size,beam_size_fwhp):
    '''
    Input: pixel size, beam size in arcmin
    Return: gaussian beam
    '''
    N=37
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) * pix_size
    X = np.outer(ones, inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.)

    beam_sigma = beam_size_fwhp / np.sqrt(8.*np.log(2))
    gaussian = np.exp(-.5 *(R/beam_sigma)**2.) / (2 * np.pi * (beam_sigma ** 2))
    gaussian = gaussian / np.sum(gaussian)

    return(gaussian)


def convolve_map_with_gaussian_beam(pix_size, beam_size_fwhp, Map):
    '''
    Input: pixel size, beam size in arcmin, image
    Return: convolved map
    '''
    gaussian = gaussian_kernal(pix_size, beam_size_fwhp)
    convolved_map = signal.fftconvolve(Map, gaussian, mode = 'same')
    
    return(convolved_map)


#def calc_scale_factor(z): #Replace with AstroPy
#    """
#    calculate the cosmic expansion scale factor
#    """
#    a = 1. + z

#     return a


#def calc_rho_critical(a,cosmo,m_sun):
#    """
#    calculate critical density
#    """
#    H_0=cosmo_h*100
#    Mpc_to_m=3.09e22
#    omega_d=1.0-(omega_m+omega_b)
#    rho_critical = (3 * H_0 ** 2)/(8 * np.pi * G) * (omega_m * a ** 3 + omega_d) / m_sun * Mpc_to_m ** 3       #msolar / MPC^3

#    return rho_critical


def calc_radius(z):
    """
    calculate radius/distance from angular diameter distance
    """
    angular_dd_z = cosmo.angular_diameter_distance(z).value * np.pi / 10800.0 
    angular_dd = cosmo.angular_diameter_distance(0.5).value * np.pi / 10800.0
    rin = 2.1 * angular_dd_z / angular_dd # Calafut et al 2021
    rout = np.sqrt(2) * rin

    return rin, rout


def radius_size(z, disk = False, ring = False):
    """
    calculcate radius related to something at image level
    """
    rin, rout = calc_radius(z)
    
    if disk:
        rows, columns = np.where(r < rin)
    elif ring:
        rows, columns = np.where(r < rout)

    value = image_size//2 - rows[0]
    
    return value


def calc_constants():
    """
    Calculate constants from input constants
    """

    constant = Thomson_sec / m_electron / (c ** 2)         #s^2/kg
    omega_d0 = 1 - omega_m
    H_0 = cosmo_h * 100 / (Mpc_to_m/1000) 
    G2 = G * 1e-6 / Mpc_to_m * m_sun               # Mpc Msun^-1 (km/s)^2 

    omega_m, omega_b0, cosmo_h, sigma8, ns = cosmo_para()
    cosmo = FlatLambdaCDM(H0=cosmo_h*100, Om0=omega_m)
    cosmo = cosmology.setCosmology('myCosmo')
    params = {'flat': True, 'H0': cosmo_h*100, 'Om0': omega_m, 'Ob0': omega_b0, 'sigma8':sigma8, 'ns': ns}
    cosmology.addCosmology('myCosmo', **params)
    
    return

