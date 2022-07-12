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

def battaglia_profile(r, Mvir, z):
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
    M200, R200, c200 = mass_adv.changeMassDefinitionCModel(Mvir/cosmo_h, z, 'vir', '200c', c_model = 'ishiyama21')
    M200 *= cosmo_h
    R200 = R200 / 1000 * cosmo_h
    
    R200 *= (1. + z)                            # Proper distance to Comoving distance
    x = r / R200
    G2 = G * 1e-6 / Mpc_to_m * m_sun                               # Mpc Msun^-1 (km/s)^2 
    gamma = -0.3
    P200 =  G2 * M200 * 200. * rho_critical * (omega_b0 / omega_m0) / 2. / (R200 / (1. + z))    # Msun km^2 / s^2 / Mpc^3

    P0 = 18.1 * ((M200 / 1e14)**0.154 * (1. + z)**-0.758)
    xc = 0.497 * ((M200 / 1e14)**-0.00865 * (1. + z)**0.731)
    beta = 4.35 * ((M200 / 1e14)**0.0393 * (1. + z)**0.415) 
    pth = P200 * P0 * (x / xc)**gamma * (1. + (x/xc))**(-1. * beta)      # Msun km^2 / s^2 / Mpc^3

    j_to_kev = 6.242e15

    pth *= (m_sun * 1e6 * j_to_kev  / ((Mpc_to_m*100)**3))       # keV/cm^3
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

def plot_img(image, z, opt = 0, path = None):
    '''
    Input: image, mode (option of 0.5/5 Mpc, default to 0.5), cmb (option of y/delta_T, default to y)
    Return: angular scale
    '''
    values = [0, 9, 18, 27, 36]
    x_label_list = [9, 4.5, 0, 4.5, 9]
    y_label_list = np.around(arcmin_to_Mpc(x_label_list, z), decimals = 2)
    if opt == 0 or opt == 3:
        option = 'hot'
        title = 'Y'
        cbar_label = r'$Y$'
    if opt == 1:
        option = 'ocean'
        title = 'T'
        cbar_label = r'$uK$'
    if opt == 2:
        option = 'viridis'
        title = 'Kernal'
        cbar_label = r' '

    fig, ax = plt.subplots(1,1)
    img = ax.imshow(image, cmap=option)
    ax.set_xticks(values)
    ax.set_xticklabels(x_label_list)
    ax.set_yticks(values)
    ax.set_yticklabels(y_label_list)
    cbar = fig.colorbar(img)
    cbar.ax.set_ylabel(cbar_label)
    plt.title(title)
    plt.xlabel('arcmin')
    plt.ylabel(r'Mpc')

    if opt == 3:
        circle_disk = plt.Circle((18, 18), radius_size(disk = True), color='green', fill=False, linewidth=2)
        circle_ring = plt.Circle((18, 18), radius_size(ring = True), color='black', fill=False, linewidth=2)
        ax.add_patch(circle_disk)
        ax.add_patch(circle_ring)

    plt.savefig(path)
    
def plot_y(r, y, z, path):
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
    plt.savefig(path)
    
def generate_img(radius, profile, f, noise_level, beam_size, z, nums, p = None, AP = False):
    if 1 in nums:
        pa = p + '1' + '.png'
        plot_y(radius, profile, z, pa)
    
    y_img = make_proj_image_new(radius,profile,extrapolate=True)

    if 2 in nums:
        pa = p + '2' + '.png'
        plot_img(y_img, z, path = pa)
    
    if 3 in nums:
        pa = p + '3' + '.png'
        gaussian = gaussian_kernal(0.5, beam_size)
        plot_img(gaussian, z, opt = 2, path = pa)

    y_con = convolve_map_with_gaussian_beam(0.5, beam_size , y_img)
    
    
    t_cmb = 2.725            #K
    fsz = f_sz(f, t_cmb)
    cmb_img = y_con * fsz * t_cmb * 1e6
    
    noise = np.random.normal(0, 1, (37, 37)) * noise_level
    CMB_noise = cmb_img + noise
    
    y_noise = CMB_noise / fsz / t_cmb / 1e6
    
    if 4 in nums:
        pa = p + '4' + '.png'
        plot_img(y_con, z, path = pa)
    if 5 in nums:
        pa = p + '5' + '.png'
        plot_img(cmb_img, z, opt = 1, path = pa)
    if 6 in nums:
        pa = p + '6' + '.png'
        plot_img(noise, z, opt = 1, path = pa)
    if 7 in nums:
        pa = p + '7' + '.png'
        plot_img(CMB_noise, z, opt = 1, path = pa)
    if 8 in nums:
        pa = p + '8' + '.png'
        plot_img(y_noise, z, path = pa)
    if 9 in nums:
        pa = p + '9' + '.png'
        plot_img(y_noise, z, opt = 3, path = pa)

    if AP:
        print("tSZ Signal: " + str(tSZ_signal(y_noise)))

def tSZ_signal(Map):
    rin = 2.1
    rout = np.sqrt(2) * rin
    
    image_size = 37
    pixel_scale = 0.5
    x,y=np.meshgrid(np.arange(image_size),np.arange(image_size))
    r = np.sqrt((x-image_size//2)**2+(y-image_size//2)**2)*pixel_scale

    # tSZ signal calculation
    disk_mean = Map[r < rin].mean()
    ring_mean = Map[(r >= rin) & (r < rout)].mean()
    tSZ = disk_mean - ring_mean
    
    return tSZ

def radius_size(disk = False, ring = False):
    rin = 2.1
    rout = np.sqrt(2) * rin
    
    image_size = 37
    pixel_scale = 0.5
    x,y=np.meshgrid(np.arange(image_size),np.arange(image_size))
    r = np.sqrt((x-image_size//2)**2+(y-image_size//2)**2)*pixel_scale

    if disk:
        rows, columns = np.where(r < rin)
        value = image_size//2 - rows[0]
        return value
    
    if ring:
        rows, columns = np.where(r < rout)
        value = image_size//2 - rows[0]
        return value

    