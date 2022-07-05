import numpy as np
import os
from scipy import interpolate
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, interp1d
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
    return omega_m, omega_b, h

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

def rho_cri(z, cosmo_h, omega_m):
    '''
    Return Unit: msolar / MPC
    '''
    a = 1. + z
    omega_d = 1 - omega_m
    H_0 = cosmo_h * 100 / (3.09e19)        #Unit of 1/s
    return (3 * H_0 ** 2)/(8 * np.pi * G) * (omega_m * a ** 3 + omega_d) / m_sun * Mpc_to_m ** 3

def battaglia_profile(r, Mvir, z, rho_critical, omega_b0, omega_m0, cosmo_h):
    '''
    Using Battaglia et al (2012). Eq. 10. 
    Input: M200 in solar mass and R200 in Mpc
    Return: Pressure profile in keV/cm^3 at radius r
    '''
    params = {'flat': True, 'H0': 70, 'Om0': 0.27, 'Ob0': 0.044, 'sigma8': 0.80, 'ns': 0.95}
    cosmology.addCosmology('myCosmo', **params)
    cosmo = cosmology.setCosmology('myCosmo')
    #Option to customize concentration, currently default, using Bullock et al. (2001)
    #cvir = concentration.concentration(Mvir, 'vir', z, model = 'ishiyama21')      #Ishiyama et al. (2021)
    M200, R200, c200 = mass_adv.changeMassDefinitionCModel(Mvir/cosmo_h, z, 'vir', '200c')
    M200 *= cosmo_h
    R200 = R200 / 1000 * cosmo_h
    
    R200 *= (1. + z)                            # Proper distance to Comoving distance
    x = r / R200
    msolar = 1.9889e30                          # kg
    mpc2cm = 3.0856e24
    G = 4.3e-9                                  # Mpc Mo^-1 (km/s)^2 
    alpha = 1.0
    gamma = -0.3
    P200 = 200. * rho_critical * omega_b0 * G * M200 / omega_m0 / 2. / (R200 / (1. + z))    # Msun km^2 / Mpc^3 / s^2

    P0 = 18.1 * ((M200 / 1e14)**0.154 * (1. + z)**-0.758)
    xc = 0.497 * ((M200 / 1e14)**-0.00865 * (1. + z)**0.731)
    beta = 4.35 * ((M200 / 1e14)**0.0393 * (1. + z)**0.415) 
    pth = P200 * P0 * (x / xc)**gamma * (1. + (x/xc))**(-1. * beta)      # Msun km^2 / Mpc^3 / s^2

    pth *= (msolar * 6.24e15 * 1e6 / mpc2cm**3)       # keV/cm^3
    p_e = pth * 0.518       # Vikram et al (2016)
    return p_e

def epp_to_y(profile):
    '''
    Input: Electron pressure profile
    Return: Compton-y profile
    '''
    constant = Thomson_sec / m_electron / (c ** 2)         #s^2/kg
    new_battaglia = profile * kevcm_to_jm
    y_pro = new_battaglia * constant * Mpc_to_m
    return y_pro

def make_proj_image_new(radius, profile,maxRadius=0.5,pixel_scale=0.005,extrapolate=False):
    '''
    Input: Profile as function of Radius, maxRadius (default to 0.5) & pixel_scale (default to 0.005) in Mpc
    Return: 2D profile
    '''
    image_size = np.int_(np.round(maxRadius*2/pixel_scale)/2+1)

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
    cosmo = FlatLambdaCDM(H0=70, Om0=0.27)
    Kpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).value
    Mpc_per_arcmin = Kpc_per_arcmin/1000
    return r / Mpc_per_arcmin


def convolve_map_with_gaussian_beam(pix_size,beam_size_fwhp,Map):
    gaussian = make_2d_gaussian_beam(pix_size,beam_size_fwhp)
  
    FT_gaussian = np.fft.fft2(np.fft.fftshift(gaussian)) # first add the shift so that it is central
    FT_Map = np.fft.fft2(np.fft.fftshift(Map)) #shift the map too
    convolved_map = np.fft.fftshift(np.real(np.fft.ifft2(FT_gaussian*FT_Map))) 
    
    return(convolved_map)
  ###############################   

def make_2d_gaussian_beam(pix_size,beam_size_fwhp):
    N=101
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) * pix_size
    X = np.outer(ones,inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.)
  
    beam_sigma = beam_size_fwhp / np.sqrt(8.*np.log(2))
    gaussian = np.exp(-.5 *(R/beam_sigma)**2.)
    gaussian = gaussian / np.sum(gaussian)
    return(gaussian)

def plot_img(image, z, mode = 1, cmb = 0, save = False, path = None):
    '''
    Input: image, mode (option of 0.5/5 Mpc, default to 0.5), cmb (option of y/delta_T, default to y)
    Return: angular scale
    '''
    if mode:    # 0.5 Mpc
        values = [-0.4, -0.2, 0, 0.2, 0.4]
        e = [-0.5, 0.5, -0.5, 0.5]
    else:   # 5 Mpc
        values = [-4, -2, 0, 2, 4]
        e = [-5, 5, -5, 5]
    if cmb:
        option = 'ocean'
        title = '$\Delta$T'
        cbar_label = r'$uK$'
    else:
        option = 'hot'
        title = 'Y'
        cbar_label = r'$Y$'
        if mode == 0:
            cbar_label = r'$\log_{10}Y$'
    fig, ax = plt.subplots(1,1)
    img = ax.imshow(image, cmap=option, extent = e)
    x_label_list = Mpc_to_arcmin(values, z)
    x_label_list = np.around(x_label_list, decimals = 2)
    ax.set_xticks(values)
    ax.set_xticklabels(x_label_list)
    cbar = fig.colorbar(img)
    cbar.ax.set_ylabel(cbar_label)
    plt.title(title)
    plt.xlabel('arcmin')
    plt.ylabel(r'Mpc')
    if save:
        plt.savefig(path)
    
def plot_y(r, y, z):
    '''
    Input: profile as function of radius
    Return: visulization (non-log & log scale)
    '''
    fig,ax = plt.subplots(1,2,figsize = (12,5))
    plt.subplots_adjust(wspace = 0.3)
    ax[0].plot(r, y, color = "red", label = "non-log")
    ax[0].set_xlabel("Mpc")
    ax[0].set_ylabel(r'Mpc$^{-1}$')
    ax[0].title.set_text("Y z="+str(z))
    ax[1].loglog(r, y, color = "blue", label = "log")
    ax[1].set_xlabel("Mpc")
    ax[1].set_ylabel(r'Mpc$^{-1}$')
    ax[1].title.set_text("Y(Log) z="+str(z))
    plt.show()
    
def generate_img(radius, profile, f, noise_level, beam_size, z, plain_y = False, Mpc5 = False, Mpc = False, cmb = False, cmb_n = False, y_n = False, s = False, p = None):
    if plain_y:
        plot_y(radius, profile, z)
    if Mpc5:
        log_image = make_proj_image_new(radius,np.log10(profile),maxRadius=5,pixel_scale=0.05,extrapolate=True)
        plot_img(log_image, z, mode = 0, save = s, path = p)
    
    y_img = make_proj_image_new(radius,profile,extrapolate=True)
    
    t_cmb = 2.725            #K
    fsz = f_sz(f, t_cmb)
    cmb_img = y_img * fsz * t_cmb * 1e6
    
    noise = np.random.normal(0, 1, (101,101)) * noise_level
    CMB_noise = cmb_img + noise
    
    y_noise = CMB_noise / fsz / t_cmb / 1e6
    
    if Mpc:
        plot_img(y_img, z, save = s, path = p)
    if cmb:
        plot_img(cmb_img, z, cmb = 1, save = s, path = p)
    if cmb_n:
        plot_img(CMB_noise, z, cmb = 1, save = s, path = p)
    if y_n:
        final = convolve_map_with_gaussian_beam(Mpc_to_arcmin(0.005, z), beam_size , y_noise)
        plot_img(final, z, save = s, path = p)