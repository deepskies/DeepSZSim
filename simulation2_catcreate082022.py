import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
from astropy.cosmology import FlatLambdaCDM
from colossus.cosmology import cosmology
from colossus.halo import mass_adv
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
from yaml.loader import SafeLoader
mpl.style.use('default')

# Constants
G = 6.6743e-11        #m^3/Kg/s^2
m_sun = 1.98847e30       #Kg
Thomson_sec = 6.65246e-29       #m^2
m_electron = 9.11e-31        #Kg
c = 299792458     #m/s
Mpc_to_m = 3.09e22
kevcm_to_jm = 1.6e-16 * 1e6

# Load image parameters from the generate.yaml file
def img_para(num):
    with open('generate.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
        # Parameters for image 1 or 2
        data = data[num]
        mass = data['mass']
        z = data['redshift']
        tele = data['telescope']
        f = data['frequency']
    return mass, z, tele, f

# Load comological parameters from the config.yaml file
def cosmo_para():
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
        cosmo = data['COSMOLOGY']
        omega_m = cosmo['Omega_m0']
        omega_b = cosmo['Omega_b0']
        h = cosmo['cosmo_h']
        sigma8 = cosmo['sigma8']
        ns = cosmo['ns']
    return omega_m, omega_b, h, sigma8, ns

# Load telescope parameters from the config.yaml file
def tele_para(tele, fre):
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
        # Parameters for type of telescope ACT_DR4/ACT_DR5/SPT
        telescope = data[tele]
        telescope_f = telescope[str(fre)+'GHz']
        beam_size = telescope_f['beam_size']
        noise_level = telescope_f['noise_level']
    return beam_size, noise_level

def battaglia_profile(r, M500, z):
    '''
    Using Battaglia et al (2012). Eq. 10. 
    Input: Virial Mass in solar mass and Radius in Mpc
    Return: Pressure profile in keV/cm^3 at radius r
    '''
    omega_m0, omega_b0, cosmo_h, sigma8, ns = cosmo_para()

    a = 1. + z
    omega_d0 = 1 - omega_m0
    H_0 = cosmo_h * 100 / (Mpc_to_m/1000) 
    rho_critical = (3 * H_0 ** 2)/(8 * np.pi * G) * (omega_m0 * a ** 3 + omega_d0) / m_sun * Mpc_to_m ** 3       #msolar / MPC^3

    params = {'flat': True, 'H0': cosmo_h*100, 'Om0': omega_m0, 'Ob0': omega_b0, 'sigma8':sigma8, 'ns': ns}
    cosmology.addCosmology('myCosmo', **params)
    cosmo = cosmology.setCosmology('myCosmo')
    #Option to customize concentration, currently default, using Bullock et al. (2001)
    #cvir = concentration.concentration(Mvir, 'vir', z, model = 'ishiyama21')      #Ishiyama et al. (2021)
    M200, R200, c200 = mass_adv.changeMassDefinitionCModel(M500/cosmo_h, z, '500c', '200c', c_model = 'ishiyama21')
    M200 *= cosmo_h
    R200 = R200 / 1000 * cosmo_h
    
    R200 *= (1. + z)                            # Proper distance to Comoving distance
    G2 = G * 1e-6 / Mpc_to_m * m_sun                               # Mpc Msun^-1 (km/s)^2 
    gamma = -0.3
    P200 =  G2 * M200 * 200. * rho_critical * (omega_b0 / omega_m0) / 2. / (R200 / (1. + z))    # Msun km^2 / s^2 / Mpc^3

    P0 = 18.1 * ((M200 / 1e14)**0.154 * (1. + z)**-0.758)
    xc = 0.497 * ((M200 / 1e14)**-0.00865 * (1. + z)**0.731)
    beta = 4.35 * ((M200 / 1e14)**0.0393 * (1. + z)**0.415) 

    x = r / R200
    pth = P200 * P0 * (x / xc)**gamma * (1. + (x/xc))**(-1. * beta)      # Msun km^2 / s^2 / Mpc^3

    j_to_kev = 6.242e15

    pth *= (m_sun * 1e6 * j_to_kev  / ((Mpc_to_m*100)**3))       # keV/cm^3
    p_e = pth * 0.518       # Vikram et al (2016)
    return p_e, M200, R200, c200

def epp_to_y(profile):
    '''
    Input: Electron pressure profile
    Return: Compton-y profile
    '''
    constant = Thomson_sec / m_electron / (c ** 2)         #s^2/kg
    new_battaglia = profile * kevcm_to_jm
    y_pro = new_battaglia * constant * Mpc_to_m
    return y_pro

def make_proj_image_new(radius, profile,range=18,pixel_scale=0.5,extrapolate=False):
    '''
    Input: Profile as function of Radius, range (default to 18) & pixel_scale (default to 0.5) in Mpc
    Return: 2D profile
    '''
    image_size = range/pixel_scale+1

    if extrapolate:
        profile_spline = interp1d(radius, profile, kind = 3, bounds_error=False, fill_value="extrapolate")
    else:
        profile_spline = interp1d(radius, profile, bounds_error=False)

    x,y=np.meshgrid(np.arange(image_size),np.arange(image_size))
    r = np.sqrt((x-image_size//2)**2+(y-image_size//2)**2)*pixel_scale
    image = profile_spline(r)
    return image
    
def f_sz(f, T_CMB):
    '''
    Input: Observation frequency f, Temperature of cmb T_CMB
    Return: Radiation frequency
    '''
    planck_const = 6.626e-34         #m^2 kg/s
    boltzman_const = 1.38e-23
    x = planck_const * f / boltzman_const / T_CMB
    return x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4

def Mpc_to_arcmin(r, z):
    '''
    Input: distance r, redshift
    Return: angular scale
    '''
    omega_m0, omega_b0, cosmo_h, sigma8, ns = cosmo_para()
    cosmo = FlatLambdaCDM(H0=cosmo_h*100, Om0=omega_m0)
    Kpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).value
    Mpc_per_arcmin = Kpc_per_arcmin/1000.
    return r / Mpc_per_arcmin

def arcmin_to_Mpc(r, z):
    '''
    Reverse of Mpc_to_arcmin
    '''
    omega_m0, omega_b0, cosmo_h, sigma8, ns = cosmo_para()
    cosmo = FlatLambdaCDM(H0=cosmo_h*100, Om0=omega_m0)
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
    X = np.outer(ones,inds)
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
    
def generate_img(radius, profile, f, noise_level, beam_size):    
    #2d array of unconvolved submap (Compton)
    y_img = make_proj_image_new(radius,profile,extrapolate=True)
    
    #gaussian kernal
    gaussian = gaussian_kernal(0.5, beam_size)

    #2d array of convolved submap (Compton)
    y_con = convolve_map_with_gaussian_beam(0.5, beam_size , y_img)
    
    t_cmb = 2.725            #K
    fsz = f_sz(f, t_cmb)

    #2d array of convolved submap (Temperature Decrement)
    cmb_img = y_con * fsz * t_cmb * 1e6
    
    #2d array of the noise
    noise = np.random.normal(0, 1, (37, 37)) * noise_level

    #Temperature Decrement with noise
    cmb_noise = cmb_img + noise
    
    #Compton with noise
    y_noise = cmb_noise / fsz / t_cmb / 1e6

    return y_img, y_con, cmb_img, noise, cmb_noise, y_noise    