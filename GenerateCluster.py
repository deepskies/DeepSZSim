import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
from astropy.cosmology import FlatLambdaCDM
from colossus.cosmology import cosmology
from colossus.halo import mass_adv
import utils




class GenerateCluster():

    def __init__():

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

        return image, x, y, r


    def tSZ_signal(z, Map):
        """
        https://kbarbary-astropy.readthedocs.io/en/latest/_modules/astropy/cosmology/funcs.html#kpc_proper_per_arcmin
        """

        rin, rout = calc_radius(z)

        disk_mean = Map[r < rin].mean()
        ring_mean = Map[(r >= rin) & (r < rout)].mean()
        tSZ = disk_mean - ring_mean
        
        return tSZ, rin


     def f_sz(f, T_CMB):
        '''
        Input: Observation frequency f, Temperature of cmb T_CMB
        Return: Radiation frequency
        '''

        x = planck_const * f / boltzman_const / T_CMB
        fsz = x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4

        return fsz


    def radius_size(z, disk = False, ring = False):

        rin, rout = calc_radius(z)
        
        if disk:
            rows, columns = np.where(r < rin)
        elif ring:
            rows, columns = np.where(r < rout)

        value = image_size//2 - rows[0]
        
        return value


     def epp_to_y(profile):
        '''
        Input: Electron pressure profile
        Return: Compton-y profile
        '''

        new_battaglia = profile * kevcm_to_jm
        y_pro = new_battaglia * constant * Mpc_to_m

        return y_pro


    def battaglia_profile(r, Mvir, z):
        '''
        Using Battaglia et al (2012). Eq. 10. 
        Input: Virial Mass in solar mass and Radius in Mpc
        Return: Pressure profile in keV/cm^3 at radius r
        '''

        a = calc_scale_factor(z)
        rho_crit = calc_rho_critical(a)

        M200, R200, c200 = mass_adv.changeMassDefinitionCModel(Mvir/cosmo_h, z, 'vir', '200c', c_model = 'ishiyama21')
        #M200, R200, c200 = mass_adv.changeMassDefinitionCModel(Mvir/cosmo_h, z, 'vir', '200c', c_model = 'ishiyama21')
        #M200, R200, c200 = mass_adv.changeMassDefinitionCModel(M500/cosmo_h, z, '500c', '200c', c_model = 'ishiyama21')
        #cvir = concentration.concentration(Mvir, 'vir', z, model = 'ishiyama21')      #Ishiyama et al. (2021)
        #Option to customize concentration, currently default, using Bullock et al. (2001)

        M200 *= cosmo_h
        R200 = R200 / 1000 * cosmo_h * (1.+z)
        x = r / R200
        gamma = -0.3
        P200 =  G2 * M200 * 200. * rho_critical * (omega_b0 / omega_m0) / 2. / (R200 / (1. + z))    # Msun km^2 / s^2 / Mpc^3

        P0 = 18.1 * ((M200 / 1e14)**0.154 * (1. + z)**-0.758)
        xc = 0.497 * ((M200 / 1e14)**-0.00865 * (1. + z)**0.731)
        beta = 4.35 * ((M200 / 1e14)**0.0393 * (1. + z)**0.415) 
        pth = P200 * P0 * (x / xc)**gamma * (1. + (x/xc))**(-1. * beta)      # Msun km^2 / s^2 / Mpc^3

        pth *= (m_sun * 1e6 * j_to_kev  / ((Mpc_to_m*100)**3))       # keV/cm^3
        p_e = pth * 0.518       # Vikram et al (2016)

        return p_e, M200, R200, c200


    def generate_cluster(radius, profile, f, noise_level, beam_size, z, nums, p = None):

        y_con = convolve_map_with_gaussian_beam(0.5, beam_size , y_img)
        fsz = f_sz(f, t_cmb)
        cmb_img = y_con * fsz * t_cmb * 1e6
        noise = np.random.normal(0, 1, (37, 37)) * noise_level
        CMB_noise = cmb_img + noise
        y_noise = CMB_noise / fsz / t_cmb / 1e6
        y_img = make_proj_image_new(radius,profile,extrapolate=True)

        if 1 in nums:
            pa = p + '1' + '.png'
            plot_y(radius, profile, z, pa)

        if 2 in nums:
            pa = p + '2' + '.png'
            plot_img(y_img, z, path = pa)
        
        if 3 in nums:
            pa = p + '3' + '.png'
            gaussian = gaussian_kernal(0.5, beam_size)
            plot_img(gaussian, z, opt = 2, path = pa)
        
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
            plot_img(y_noise, z, opt = 3, path = pa) #vizualization starts working from z = 0.115

        #return tSZ_signal(z, y_con), tSZ_signal(z, y_noise)
        return y_img, y_con, cmb_img, noise, cmb_noise, y_noise, SZsignal, aperture  

    
        
   

   
 
        
    
        