import numpy as np
import simsz.utils as utils
from scipy.interpolate import interp1d
from colossus.cosmology import cosmology
from colossus.halo import mass_adv
from pixell import enmap, powspec, enplot
import camb

from astropy import constants as c
from astropy import units as u
import simsz.simtools

class GenerateCluster():

    def __init__(self):
        self.data =[]


    def P200_Battaglia2012(self,cosmo,redshift_z,M200_SM,R200_mpc):
        '''
        Calculates the P200 pressure profile of a cluster, as defined in 
        Battaglia 2012

        Parameters:
        -----------
        cosmo: FlatLambaCDM instance
            background cosmology for density calculation
        redshift_z: float
            the redshift of the cluster (unitless)
        M200_SM: float
            the mass contained within R200, in units of solar masses
        R200_mpc: float
            the radius of the cluster at 200 times the critical density of the 
            universe in units of Mpc

        Returns:
        --------
        P200_kevcm3: Quantity instance
            the thermal pressure profile of the shell defined by R200 in units 
            of keV/cm**3
        '''

        GM200 = c.G * M200_SM*u.Msun * 200. * cosmo.critical_density(redshift_z)
        fbR200 = (cosmo.Ob0/cosmo.Om0) / (2. * R200_mpc*u.Mpc) #From Battaglia2012
        P200 = GM200 * fbR200
        P200_kevcm3=P200.to(u.keV/u.cm**3.) #Unit conversion to keV/cm^3

        return(P200_kevcm3)

    def param_Battaglia2012(self,A0,alpha_m,alpha_z,M200_SM,redshift_z):
        '''
        Calculates independent params as using the formula from Battaglia 2012, 
        Equation 11 in order for use in the pressure profile defined in 
        Equation 10

        Parameters:
        -----------
        A0: float
            normalization factor
        alpha_m: float
            power law index for the mass-dependent part of the function
        alpha_z: float
            power law index for the redshift dependent part
        M200_SM: float
            the mass of the cluster at 200 times the critical density of the 
            universe, in units of solar masses
        redshift_z: float
            the redshift of the cluster (unitless)

        Returns:
        --------
        A: float
            the parameter A given the formula from Eq 11
        '''

        A = A0 * (M200_SM/1e14)**alpha_m * (1.+redshift_z)**alpha_z

        return(A)

    def Pth_Battaglia2012(self,radius_mpc,R200_mpc,gamma,alpha,beta,xc,P0):
        '''
        Calculates Pth using the battaglia fit profile, Battaglia 2012, 
        Equation 10 Pth is the thermal pressure profile normalized over P200

        Parameters:
        -----------
        radius_mpc: float
            the radius for the pressure to be calculated at, in units of Mpc
        R200_mpc: float
            the radius of the cluster at 200 times the critical density of the 
            universe, in units of Mpc
        gamma: float
            fixed paremeter defined by Battaglia 2012
        alpha: float
            fixed parameter defined by Battaglia 2012
        beta: float
            power law index
        xc: float
            fits for the core-scale
        P0: float
            the normalization factor/amplitude,

        Returns:
        --------
        Pth: float
            the thermal pressure profile normalized over P200, units of 
            keV/cm**3
        '''

        x=radius_mpc/R200_mpc

        Pth = P0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-beta)

        return(Pth)

    def epp_to_y(self, profile, radii_mpc, P200_kevcm3, R200_mpc, **kwargs):
        '''
        Converts from an electron pressure profile to a compton-y profile,
        integrates over line of sight from -1 to 1 Mpc relative to center.

        Parameters:
        -----------
        profile:
            Method to get thermal pressure profile in Kev/cm^3, accepts radius, 
            R200 and **kwargs
        radii_mpc: array
            the array of radii corresponding to the profile in Mpc
        P200_kevcm3: float
            as defined in battaglia2012, needed for normalization of Battaglia 
            profile, units of keV/cm**3
        R200_mpc: float
            the radius of the cluster at 200 times the critical density of the 
            universe in units of Mpc

        Return:
        -------
        y_pro: array
            Compton-y profile corresponding to the radii
        '''
        radii_mpc = radii_mpc * u.Mpc
        pressure_integ = np.empty(radii_mpc.size)
        i = 0
        l_mpc = np.linspace(0,R200_mpc,10000) # Get line of sight axis
        l = l_mpc * (1 * u.Mpc).to(u.m).value # l in meters
        keVcm_to_Jm = (1 * u.keV/(u.cm**3.)).to(u.J/(u.m**3.))
        thermal_to_electron_pressure = 1/1.932 # from Battaglia 2012, assumes 
                                                #fully ionized medium
        for radius in radii_mpc:
            #Multiply profile by P200 specifically for Battaglia 2012 profile, 
            # since it returns Pth/P200 instead of Pth
            th_pressure = profile(np.sqrt(l_mpc**2 + radius.value**2), 
                                  R200_mpc=R200_mpc, **kwargs)
            th_pressure = th_pressure * P200_kevcm3.value #pressure as a 
            #                                               function of l
            th_pressure = th_pressure * keVcm_to_Jm.value # Use multiplication 
            #                           by a precaluated factor for efficiency
            pressure = th_pressure* thermal_to_electron_pressure
            integral = np.trapz(pressure, l) * 2 #integrate over pressure in 
            #J/m^3 to get J/m^2, multiply by factor of 2 to get from -R200 to 
            # R200 (assuming spherical symmetry)
            pressure_integ[i] = integral
            i += 1
        y_pro = pressure_integ * c.sigma_T.value/ (c.m_e.value * c.c.value**2)
        return y_pro

    def make_y_submap(self, profile, redshift_z, cosmo, width, pix_size_arcmin, 
                      *args, **kwargs):
        '''
        Converts from an electron pressure profile to a compton-y profile,
        integrates over line of sight from -1 to 1 Mpc relative to center.

        Parameters:
        -----------
        profile:
            Method to get thermal pressure profile in Kev/cm^3, accepts radius,
              R200 and **kwargs
        redshift_z: float
            the redshift of the cluster (unitless)
        cosmo: FlatLambaCDM instance
            background cosmology for density calculation
        width: float
            num pixels to each side of center; end shape of submap will be 
            (2*width +1, 2*width +1)
        pix_size_arcmin: float
            size of each pixel in arcmin

        Return:
        -------
        y_map: array
            Compton-y submap with shape (2*width +1, 2*width +1)
        '''
        X = np.arange(-width, width + pix_size_arcmin, pix_size_arcmin)
        X = utils.arcmin_to_Mpc(X, redshift_z, cosmo)
        #Solves issues of div by 0
        X[(X<=pix_size_arcmin/10)&(X>=-pix_size_arcmin/10)] = pix_size_arcmin/10
        Y = np.transpose(X)

        R = []
        y_map = np.empty((X.size, Y.size))

        for i in X:
            for j in Y:
                R.append(np.sqrt(i**2 + j**2))

        R = np.array(R)
        cy = self.epp_to_y(profile, R, **kwargs) #evaluate compton-y for each 
        #                                           neccesary radius

        for i in range(X.size):
            for j in range(Y.size):
                y_map[i][j] = cy[np.where(
                    R == np.sqrt(X[i]**2 + Y[j]**2))[0]][0] 
                # assign the correct compton-y to the radius

        return y_map


    def generate_y_submap(self, redshift_z, M200_SM, R200_mpc, cosmo, 
                          width, pix_size_arcmin, profile="Battaglia2012"):
        '''
        Converts from an electron pressure profile to a compton-y profile,
        integrates over line of sight from -1 to 1 Mpc relative to center.

        Parameters:
        ----------
        redshift_z: float
            the redshift of the cluster (unitless)
        M200_SM:
            the mass contained within R200 in solar masses
        R200_mpc: float
            the radius of the cluster at 200 times the critical density of the 
            universe in Mpc
        cosmo: FlatLambaCDM instance
            background cosmology for density calculation
        width: float
            num pixels to each side of center; end shape of submap will be 
            (2*width +1, 2*width +1)
        pix_size_arcmin: float
            size of each pixel in arcmin
        profile: str
            Name of profile, currently only supports "Battaglia2012"

        Return:
        ------
        y_map: array
            Compton-y submap with dimension (2*width +1 , 2*width +1)
        '''
        if profile != "Battaglia2012":
            return None

        P200 = self.P200_Battaglia2012(cosmo,redshift_z,M200_SM,
                                       R200_mpc) #P200 from Battaglia 2012
        P0=self.param_Battaglia2012(18.1,0.154,-0.758,M200_SM,
                                    redshift_z) #Parameter computation from 
                                                #Table 1 Battaglia et al. 2012
        xc=self.param_Battaglia2012(0.497,-0.00865,0.731,M200_SM,
                                    redshift_z)
        beta=self.param_Battaglia2012(4.35,0.0393,0.415,M200_SM,
                                      redshift_z)
        y_map = self.make_y_submap(self.Pth_Battaglia2012, redshift_z, cosmo, 
                                   width, pix_size_arcmin, R200_mpc=R200_mpc, 
                                   gamma=-0.3,alpha=1.0,beta=beta,xc=xc,P0=P0, 
                                   P200_kevcm3=P200)

        return y_map
# 5) Generate noise map

    def generate_cluster(self, radius, profile, f_ghz, noise_level, 
                         beam_size_arcmin, redshift_z, nums, p = None): 
        """
        combine all elements to generate a cluster object

        Parameters:
        -----------
        radius: float
        profile:
        f: float
            Observation frequency f, in units of GHz
        noise_level: float
        beam_size_arcmin: float
            beam size in arcmin
        redshift_z: float
            redshift (unitless)
        nums: list or array

        Returns:
        -------
        a cluster object
        """

        y_con = convolve_map_with_gaussian_beam(0.5, beam_size_arcmin , y_img)
        fsz = f_sz(f_ghz, t_cmb)
        cmb_img = y_con * fsz * t_cmb * 1e6
        noise = np.random.normal(0, 1, (37, 37)) * noise_level
        CMB_noise = cmb_img + noise
        y_noise = CMB_noise / fsz / t_cmb / 1e6
        y_img = make_proj_image_new(radius,profile,extrapolate=True)

        if 1 in nums:
            pa = p + '1' + '.png'
            plot_y(radius, profile, redshift_z, pa)

        if 2 in nums:
            pa = p + '2' + '.png'
            plot_img(y_img, redshift_z, path = pa)

        if 3 in nums:
            pa = p + '3' + '.png'
            gaussian = gaussian_kernal(0.5, beam_size_arcmin)
            plot_img(gaussian, redshift_z, opt = 2, path = pa)

        if 4 in nums:
            pa = p + '4' + '.png'
            plot_img(y_con, redshift_z, path = pa)
        if 5 in nums:
            pa = p + '5' + '.png'
            plot_img(cmb_img, redshift_z, opt = 1, path = pa)
        if 6 in nums:
            pa = p + '6' + '.png'
            plot_img(noise, redshift_z, opt = 1, path = pa)
        if 7 in nums:
            pa = p + '7' + '.png'
            plot_img(CMB_noise, redshift_z, opt = 1, path = pa)
        if 8 in nums:
            pa = p + '8' + '.png'
            plot_img(y_noise, redshift_z, path = pa)
        if 9 in nums:
            pa = p + '9' + '.png'
            plot_img(y_noise, redshift_z, opt = 3, path = pa) #vizualization 
                                                #starts working from z = 0.115

        #return tSZ_signal(z, y_con), tSZ_signal(z, y_noise)
        return y_img, y_con, cmb_img, noise, cmb_noise, y_noise, SZsignal, aperture

    def get_r200_and_c200(self,cosmo,sigma8,ns,M200_SM,redshift_z):
        '''
        Parameters:
        ----------
        cosmo: FlatLambaCDM instance
            background cosmology
        sigma8: float
        ns: float
        M200_SM: float
            the mass contained within R200, in units of solar masses
        redshift_z: float
            redshift of the cluster (unitless)

        Returns:
        -------
        M200_SM: float
            the mass contained within R200, in units of solar masses
        R200_mpc: float
            the radius of the cluster at 200 times the critical density of the universe in Mpc
        c200: float
            concentration parameter
        '''


        params = {'flat': True, 'H0': cosmo.H0.value, 'Om0': cosmo.Om0, 'Ob0': cosmo.Ob0, 'sigma8':sigma8, 'ns': ns}
        cosmology.addCosmology('myCosmo', **params)
        cosmo_colossus= cosmology.setCosmology('myCosmo')

        M200_SM, R200_mpc, c200 = mass_adv.changeMassDefinitionCModel(M200_SM/cosmo.h, redshift_z, '200c', '200c', c_model = 'ishiyama21')
        M200_SM *= cosmo.h #From M_solar/h to M_solar
        R200_mpc = R200_mpc*cosmo.h/1000 #From kpc/h to Mpc
        R200_mpc = R200_mpc/cosmo.scale_factor(redshift_z) #From Mpc proper to Mpc comoving
        return M200_SM, R200_mpc, c200


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
