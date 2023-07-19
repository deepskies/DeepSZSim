import numpy as np
import simsz.utils as utils

from scipy.interpolate import interp1d
from colossus.halo import mass_adv

from astropy.constants import G, sigma_T, m_e, c, h, k_B
from astropy import units as u

class GenerateCluster():

    def __init__(self):
        self.data =[]
        
        
    def P200_Battaglia2012(self,cosmo,redshift_z,M200,R200):
        '''
        Calculates the P200 pressure profile of a cluster, as defined in Battaglia 2012

        Parameters:
        R200, the radius of the cluster at 200 times the critical density of the universe
        M200, the mass contained within R200
        redshift_z, the redshift of the cluster
        cosmo, background cosmology for density calculation

        Returns P200, the thermal pressure profile of the shell defined by R200
        '''
        
        P200 = G * M200*u.Msun * 200. * cosmo.critical_density(redshift_z) * (cosmo.Ob0/cosmo.Om0) / (2. * R200*u.Mpc) #From Battaglia 2012
        P200=P200.to(u.keV/u.cm**3.) #Unit conversion to keV/cm^3
        
        return(P200)
    
    def param_Battaglia2012(self,A0,alpha_m,alpha_z,M200,redshift_z):
        '''
        Calculates independent params as using the formula from Battaglia 2012, Equation 11 
        in order for use in the pressure profile defined in Equation 10

        Parameters:
        A0, normalization factor
        alpha_m, power law index for the mass-dependent part of the function
        alpha_z, power law index for the redshift dependent part
        M200, the mass of the cluster at 200 times the critical density of the universe
        redshift_z, the redshift of the cluster
        
        Returns the parameter A given the formula from Eq 11
        '''
        
        A = A0 * (M200/1e14)**alpha_m * (1.+redshift_z)**alpha_z

        return(A)
    
    def Pth_Battaglia2012(self,radius,R200,gamma,alpha,beta,xc,P0):
        '''
        Calculates Pth using the battaglia fit profile, Battaglia 2012, Equation 10
        Pth is the thermal pressure profile normalized over P200

        Parameters:
        P0 is the normalization factor/amplitude,
        xc fits for the core-scale
        beta is a power law index
        gamma is a fixed paremeter defined by Battaglia 2012
        alpha is a fixed parameter defined by Battaglia 2012
        R200, the radius of the cluster at 200 times the critical density of the universe
        radius, the raidus for the pressure to be calculated at

        Returns Pth, the thermal pressure profile normalized over P200
        '''

        x=radius/R200
        
        Pth = P0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-beta)
        
        return(Pth)
    
    def epp_to_y(self, profile, radii, P200, R200, **kwargs): 
        '''
        Converts from an electron pressure profile to a compton-y profile,
        integrates over line of sight from -1 to 1 Mpc relative to center.
        Parameters: 
        profile, Method to get thermal pressure profile in Kev/cm^3, accepts radius, R200 and **kwargs
        radii, the array of radii corresponding to the profile in Mpc
        P200, as defined in battaglia2012, needed for normalization of Battaglia profile
        R200, the radius of the cluster at 200 times the critical density of the universe

        Return: Compton-y profile corresponding to the radii
        '''
        radii = radii * u.Mpc
        pressure_integrated = np.empty(radii.size)
        i = 0
        l_mpc = np.linspace(0,R200,10000) # Get line of sight axis
        l = l_mpc * (1 * u.Mpc).to(u.m).value # l in meters
        keVcm_to_Jm = (1 * u.keV/(u.cm**3.)).to(u.J/(u.m**3.))
        thermal_to_electron_pressure = 1/1.932 # from Battaglia 2012, assumes fully ionized medium
        for radius in radii:
            #Multiply profile by P200 specifically for Battaglia 2012 profile, since it returns Pth/P200 instead of Pth
            th_pressure = profile(np.sqrt(l_mpc**2 + radius.value**2), R200=R200, **kwargs) * P200.value #pressure as a function of l, multiplication by P20 needed ad 
            th_pressure = th_pressure * keVcm_to_Jm.value # Use multiplication by a precaluated factor for efficiency
            pressure = th_pressure* thermal_to_electron_pressure
            integral = np.trapz(pressure, l) * 2 #integrate over pressure in J/m^3 to get J/m^2, multiply by factor of 2 to get from -R200 to R200 (assuming spherical symmetry)
            pressure_integrated[i] = integral
            i += 1
        y_pro = pressure_integrated * sigma_T.value/ (m_e.value * c.value**2)
        return y_pro
    
    def make_y_submap(self, profile, redshift_z, cosmo, width, pix_size, *args, **kwargs):
        '''
        Converts from an electron pressure profile to a compton-y profile,
        integrates over line of sight from -1 to 1 Mpc relative to center.
        Parameters: 
        profile, Method to get thermal pressure profile in Kev/cm^3, accepts radius, and **kwargs
        redshift_z, the redshift of the cluster
        cosmo, background cosmology for density calculation
        width, width of submap in arcmin
        pix_size, size of each pixel in arcmin

        Return: Compton-y submap
        '''
        X = np.arange(-width, width, pix_size)
        X = utils.arcmin_to_Mpc(X, redshift_z, cosmo)
        X[X==0] = 0.001
        Y = np.transpose(X)
        # radial component R
        R = []
        y_map = np.empty((X.size, Y.size))
        
        for i in X:
            for j in Y:
                R.append(np.sqrt(i**2 + j**2))
    
        R = np.array(R)
        cy = self.epp_to_y(profile, R, **kwargs)

        for i in range(X.size):
            for j in range(Y.size):
                y_map[i][j] = cy[np.where(R == np.sqrt(X[i]**2 + Y[j]**2))[0]][0]
    
        return y_map
  
    def f_sz(self, freq, T_CMB, *args, **kwargs):
        '''
        Input: Observation frequency f in GHz, Temperature of cmb T_CMB
        Return: Radiation frequency
        '''

        f=freq*u.GHz #Takes input in units of GHz
        f=f.to(1/u.s) #Unit conversion
        x = h * f / k_B / T_CMB
        fsz = x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4

        return fsz



####Functions needed in this file:
# 3) Convolve submap with beam

    
# 5) Generate noise map
    
    def generate_cluster(self, radius, profile, f, noise_level, beam_size, z, nums, p = None): #SOME OF THE ABOVE WILL BE TAKEN FROM THIS OUTDATED FUNCTION
        """
        combine all elements to generate a cluster object
        """

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
   
   

#FUNCTIONS BELOW HERE HAVE NOT BEEN TESTED OR USED RECENTLY; MIGHT BE USEFUL FOR THE ABOVE TO-DO LIST     


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


    def battaglia_profile(r, Mvir, z, cosmo): #THIS IS OLD; WILL LIKELY DELETE SOON
        '''
        Using Battaglia et al (2012). Eq. 10. 
        Input: Virial Mass in solar mass and Radius in Mpc
        Return: Pressure profile in keV/cm^3 at radius r
        '''
        
        a = cosmo.scale_factor(z)
        rho_critical = cosmo.critical_density(0.5).to(u.M_sun/u.Mpc**3) #In M_sun/Mpc^3
        
        

        M200, R200, c200 = mass_adv.changeMassDefinitionCModel(Mvir/cosmo.h, z, 'vir', '200c', c_model = 'ishiyama21')
        #M200, R200, c200 = mass_adv.changeMassDefinitionCModel(Mvir/cosmo_h, z, 'vir', '200c', c_model = 'ishiyama21')
        #M200, R200, c200 = mass_adv.changeMassDefinitionCModel(M500/cosmo_h, z, '500c', '200c', c_model = 'ishiyama21')
        #cvir = concentration.concentration(Mvir, 'vir', z, model = 'ishiyama21')      #Ishiyama et al. (2021)
        #Option to customize concentration, currently default, using Bullock et al. (2001)
 
        M200 *= cosmo.h
        R200 = R200 / 1000 * cosmo.h * (1.+z)
        x = r / R200
        
        G2 = G * 1e-6 / float(Mpc_to_m) * m_sun
        gamma = -0.3
        
        P200 =  G2 * M200 * 200. * rho_critical * (omega_b / omega_m) / 2. / (R200 / (1. + z))    # Msun km^2 / s^2 / Mpc^3

        P0 = 18.1 * ((M200 / 1e14)**0.154 * (1. + z)**-0.758)
        xc = 0.497 * ((M200 / 1e14)**-0.00865 * (1. + z)**0.731)
        beta = 4.35 * ((M200 / 1e14)**0.0393 * (1. + z)**0.415) 
        pth = P200 * P0 * (x / xc)**gamma * (1. + (x/xc))**(-1. * beta)      # Msun km^2 / s^2 / Mpc^3

        pth *= (m_sun * 1e6 * j_to_kev  / ((Mpc_to_m*100)**3))       # keV/cm^3
        p_e = pth * 0.518       # Vikram et al (2016)

        return p_e, M200, R200, c200 

    
        
   

   
 
        
    
        