"""
pressure profile, Compton-y, R200, C200 and temperature submap generating functions based on halo redshift and mass information
"""

import numpy as np
from deepszsim import utils
from colossus.cosmology import cosmology
from colossus.halo import mass_adv

from astropy import constants as c
from astropy import units as u

keVcm3_to_Jm3 = ((1 * u.keV / (u.cm**3.)).to(u.J / (u.m**3.))).value
thermal_to_electron_pressure = 1 / 1.932  # from Battaglia 2012, assumes
Mpc_to_m = (1 * u.Mpc).to(u.m).value
Thomson_scale = (c.sigma_T/(c.m_e * c.c**2)).value
# fully ionized medium

def _param_Battaglia2012(A0, alpha_m, alpha_z, M200_SM, redshift_z):
    '''
    Calculates independent params using the formula from Battaglia 2012, Equation 11
    in order for use in the pressure profile defined in Equation 10

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
        universe in units of solar masses
    redshift_z: float
        the redshift of the cluster

    Returns:
    --------
    A: float
        the parameter A given the formula from Battaglia 2012, Eq 11
    '''
    
    A = A0 * (M200_SM / 1e14)**alpha_m * (1. + redshift_z)**alpha_z
    
    return (A)


def _P0_Battaglia2012(M200_SM, redshift_z):
    """
    Calculates P0, the normalization factor/amplitude, 
    from Battaglia 2012, using the values from Table 1,

    Parameters:
    -----------
    M200_SM: float
        the mass of the cluster at 200 times the critical density of the 
        universe, in units of solar masses
    redshift_z: float
        the redshift of the cluster (unitless)

    Returns:
    -------
    P0: float
        the normalization factor for the Battaglia 2012 profile

    """
    return _param_Battaglia2012(18.1, 0.154, -0.758, M200_SM, redshift_z)


def _xc_Battaglia2012(M200_SM, redshift_z):
    """
    Calculates xc, the core-scale factor,
    from Battaglia 2012, using the values from Table 1,

    Parameters:
    -----------
    M200_SM: float
        the mass of the cluster at 200 times the critical density of the 
        universe, in units of solar masses
    redshift_z: float
        the redshift of the cluster (unitless)

    Returns:
    --------
    xc: float
        the core-scale factor for the Battaglia 2012 profile

    """
    return _param_Battaglia2012(0.497, -0.00865, 0.731, M200_SM, redshift_z)


def _beta_Battaglia2012(M200_SM, redshift_z):
    """
    Calculates beta, the power law index, 
    from Battaglia 2012, using the values from Table 1,

    Parameters:
    ----------
    M200_SM: float
        the mass of the cluster at 200 times the critical density of the 
        universe, in units of solar masses
    redshift_z: float
        the redshift of the cluster (unitless)

    Returns:
    -------
    beta: float
        the power law index for the Battaglia 2012 profile

    """
    return _param_Battaglia2012(4.35, 0.0393, 0.415, M200_SM, redshift_z)


def P200_Battaglia2012(M200_SM, redshift_z, load_vars_dict, R200_Mpc = None):
    '''
    Calculates the P200 pressure of a cluster, as defined in
    Battaglia 2012

    Parameters:
    -----------
    M200_SM: float
        the mass contained within R200, in units of solar masses
    redshift_z: float
        the redshift of the cluster (unitless)
    load_vars_dict: instance of load_vars.load_vars()
        dictionary that includes background cosmology, ns, and sigma8 (necessary for the
        calculation of R200 and c200)
    R200_Mpc: None or float
        if None, will calculate the radius that corresponds to the mass M200, the redshift redshift_z,
        and the cosmology contained in load_vars_dict

    Returns:
    --------
    P200_kevcm3: Quantity instance
        the thermal pressure of the shell defined by R200 in units
        of keV/cm**3
    '''
    cosmo = load_vars_dict['cosmo']

    if R200_Mpc is None:
        R200_Mpc = get_r200_angsize_and_c200(M200_SM, redshift_z, load_vars_dict)[1]
    
    GM200 = c.G * M200_SM * u.Msun * 200. * cosmo.critical_density(redshift_z)
    fbR200 = (cosmo.Ob0 / cosmo.Om0) / (2. * R200_Mpc * u.Mpc)  # From Battaglia2012
    P200 = GM200 * fbR200
    P200_kevcm3 = P200.to(u.keV / u.cm**3.)  # Unit conversion to keV/cm^3
    
    return (P200_kevcm3)

def _Pth_Battaglia2012(P0, radius_mpc, R200_Mpc, alpha, beta, gamma, xc):
    '''
    Calculates the Pth profile using the Battaglia profile, Battaglia 2012,
    Equation 10. Pth is unitless. It is normalized by P200

    Parameters:
    -----------
    P0: float
        the normalization factor/amplitude,
    radius_mpc: float or np.ndarray(float)
        the radius or radii for the pressure to be calculated at, in units of Mpc
    R200_Mpc: float
        the radius of the cluster at 200 times the critical density of the
        universe, in units of Mpc
    alpha: float
        fixed parameter equal to 1.0 in Battaglia 2012
    beta: float
        power law index
    gamma: float
        fixed paremeter equal to -0.3 in Battaglia 2012
    xc: float
        core-scale radius

    Returns:
    --------
    Pth: float or np.ndarray(float)
        the thermal pressure profile normalized by P200 (which itself has units of
        keV/cm**3)
    '''
    
    x = radius_mpc / R200_Mpc
    
    Pth = P0 * (x / xc)**gamma * (1 + (x / xc)**alpha)**(-beta)
    
    return (Pth)

def Pth_Battaglia2012(radius_mpc, M200_SM, redshift_z, load_vars_dict = None,
                      alpha = 1.0, gamma = -0.3, R200_Mpc = None):
    '''
    Calculates the Pth profile using the Battaglia profile, Battaglia 2012,
    Equation 10. Pth is unitless. It is normalized by P200

    Parameters:
    -----------
    radius_mpc: float or np.ndarray(float)
        the radius for the pressure to be calculated at, in units of Mpc
    M200_SM: float
        the mass contained within R200, in units of solar masses
    redshift_z: float
        the redshift of the cluster (unitless)
    load_vars_dict: instance of load_vars.load_vars()
        dictionary that includes background cosmology, ns, and sigma8 (necessary for the
        calculation of R200 and c200)
    alpha: float
        variable fixed by Battaglia et al 2012 to 1.0
    gamma: float
        variable fixed by Battaglia et al 2012 to -0.3
    R200_Mpc: None or float
        if None, will calculate the radius that corresponds to the mass M200, the redshift redshift_z,
        and the cosmology contained in load_vars_dict

    Returns:
    --------
    Pth: float or np.ndarray(float)
        the thermal pressure profile normalized by P200 (which itself has units of
        keV/cm**3)
    '''
    
    if R200_Mpc is None:
        if load_vars_dict is None:
            print("must specify either `load_vars_dict` or `R200_Mpc`")
            return None
        R200_Mpc = get_r200_angsize_and_c200(M200_SM, redshift_z, load_vars_dict)[1]
    P0 = _P0_Battaglia2012(M200_SM, redshift_z)
    xc = _xc_Battaglia2012(M200_SM, redshift_z)
    beta = _beta_Battaglia2012(M200_SM, redshift_z)
    
    return _Pth_Battaglia2012(P0, radius_mpc, R200_Mpc, alpha, beta, gamma, xc)


def Pe_to_y(profile, radii_mpc, M200_SM, redshift_z, load_vars_dict, alpha = 1.0, gamma = -0.3, R200_Mpc = None):
    '''
    Converts from an electron pressure profile to a compton-y profile,
    integrates over line of sight from -1 to 1 Mpc relative to center.

    Parameters:
    -----------
    profile: method
        Method to get thermal pressure profile, accepts radius, M200, redshift_z, cosmo,
        and two additional parameters alpha and gamma that are usually fixed
    radii_mpc: array
        the array of radii corresponding to the profile in Mpc
    M200_SM: float
        the mass contained within R200, in units of solar masses
    redshift_z: float
        the redshift of the cluster (unitless)
    load_vars_dict: instance of load_vars.load_vars()
        dictionary that includes background cosmology, ns, and sigma8 (necessary for the
        calculation of R200 and c200)
    alpha: float
        variable fixed by Battaglia et al 2012 to 1.0
    gamma: float
        variable fixed by Battaglia et al 2012 to -0.3
    R200_Mpc: None or float
        if None, will calculate the radius that corresponds to the mass M200, the redshift redshift_z,
        and the cosmology contained in load_vars_dict

    Return:
    -------
    y_pro: array
        Compton-y profile corresponding to the radii
    '''
    if R200_Mpc is None:
        R200_Mpc = get_r200_angsize_and_c200(M200_SM, redshift_z, load_vars_dict)[1]
    radii_mpc = (radii_mpc * u.Mpc).value
    if profile != "Battaglia2012":
        print("only implementing `Battaglia2012` for profile")
    profile = Pth_Battaglia2012
    pressure_integ = np.empty_like(radii_mpc)
    P200_kevcm3 = P200_Battaglia2012(M200_SM, redshift_z, load_vars_dict, R200_Mpc = R200_Mpc).value
    
    rmax = radii_mpc.max()
    
    for i, radius in enumerate(radii_mpc):
        # Multiply profile by P200 specifically for Battaglia 2012 profile,
        # since it returns Pth/P200 instead of Pth
        rv = radius
        if (rv >= R200_Mpc):
            pressure_integ[i] = 0
        else:
            l_mpc = np.linspace(0, np.sqrt(rmax**2. - rv**2.) + 1., 1000)  # Get line of sight axis
            th_pressure = profile(np.sqrt(l_mpc**2 + rv**2), M200_SM, redshift_z, load_vars_dict = None, alpha = alpha,
                                  gamma = gamma, R200_Mpc = R200_Mpc)
            integral = np.trapz(th_pressure, l_mpc)
            pressure_integ[i] = integral
    y_pro = pressure_integ * P200_kevcm3 * keVcm3_to_Jm3 * Thomson_scale * thermal_to_electron_pressure * 2 * Mpc_to_m
    return y_pro


def _make_y_submap(profile, M200_SM, redshift_z, load_vars_dict, image_size_pixels, pixel_size_arcmin, alpha = 1.0,
                   gamma = -0.3, R200_Mpc = None):
    '''
    Converts from an electron pressure profile to a compton-y profile,
    integrates over line of sight from -1 to 1 Mpc relative to center.

    Parameters:
    -----------
    profile:
        Method to get thermal pressure profile in Kev/cm^3, accepts radius,
            R200 and **kwargs
    M200_SM: float
        mass contained in R200, in units of solar masses
    redshift_z: float
        the redshift of the cluster (unitless)
    load_vars_dict: instance of load_vars.load_vars()
        dictionary that includes background cosmology, ns, and sigma8 (necessary for the
        calculation of R200 and c200)
    image_size_pixels: int
        size of final submap in number of pixels
    pixel_size_arcmin: float
        size of each pixel in arcmin
    alpha: float
        variable fixed by Battaglia et al 2012 to 1.0
    gamma: float
        variable fixed by Battaglia et al 2012 to -0.3
    R200_Mpc: None or float
        if None, will calculate the radius that corresponds to the mass M200, the redshift redshift_z,
        and the cosmology contained in load_vars_dict

    Return:
    -------
    y_map: array
        Compton-y submap with shape (image_size_pixels, image_size_pixels)
    '''

    X = np.linspace(0, (image_size_pixels // 2) * pixel_size_arcmin, image_size_pixels//2 + 1)
    X = utils.arcmin_to_Mpc(X, redshift_z, load_vars_dict['cosmo'])
    # Solves issues of div by 0
    #X[(X <= pixel_size_arcmin / 10) & (X >= -pixel_size_arcmin / 10)] = pixel_size_arcmin / 10
    mindist = utils.arcmin_to_Mpc(pixel_size_arcmin*0.1, redshift_z, load_vars_dict['cosmo'])
    R = np.maximum(mindist, np.sqrt(X[:, None]**2 + X[None, :]**2).flatten())
    
    cy = Pe_to_y(profile, R, M200_SM, redshift_z, load_vars_dict, alpha = alpha, gamma = gamma, R200_Mpc = R200_Mpc)  #
    # evaluate compton-y for each
    # neccesary radius

    y_map = np.zeros((X.size*2 - 1, X.size*2 - 1))
    for i, x in enumerate(X):
        for j in range(i, len(X)):
            y = X[j]
            ijval = cy[np.where(np.isclose(R, np.maximum(mindist, np.sqrt(x**2 + y**2)),
                                         atol=1.e-10, rtol=1.e-10))[0]][0]
            y_map[X.size + i - 1][X.size + j - 1] = ijval
            if j != i:
                y_map[X.size + j - 1][X.size + i - 1] = ijval
    for i in range(len(X)):
        y_map[X.size - i - 1] = y_map[X.size + i - 1]
    for j in range(len(X)):
        y_map[:, X.size - j - 1] = y_map[:, X.size + j - 1]
    # assign the correct compton-y to the radius
    
    return y_map


def generate_y_submap(M200_SM, redshift_z, profile = "Battaglia2012",
                      image_size_pixels = None, pixel_size_arcmin = None, load_vars_dict = None, alpha = 1.0, gamma = -0.3,
                      R200_Mpc = None):
    '''
    Converts from an electron pressure profile to a compton-y profile,
    integrates over line of sight from -1 to 1 Mpc relative to center.

    Parameters:
    ----------
    M200_SM:
        the mass contained within R200 in solar masses
    redshift_z: float
        the redshift of the cluster (unitless)
    profile: str
        name of profile, currently only supports "Battaglia2012"
    image_size_pixels: float
        num pixels to each side of center; end shape of submap will be 
        (image_size_pixels, image_size_pixels)
    pixel_size_arcmin: float
        size of each pixel in arcmin
    load_vars_dict: dict
        result of running the load_vars() function, which includes a dictionary of cosmological and experimental
        parameters
    alpha: float
        variable fixed by Battaglia et al 2012 to 1.0
    gamma: float
        variable fixed by Battaglia et al 2012 to -0.3
    R200_Mpc: None or float
        if None, will calculate the radius that corresponds to the mass M200, the redshift redshift_z,
        and the cosmology contained in load_vars_dict

    Return:
    ------
    y_map: array
        Compton-y submap with dimension (image_size_pixels, image_size_pixels)
    '''
    if profile != "Battaglia2012":
        print("only implementing Battaglia2012")
        return None
    
    if load_vars_dict is not None:
        image_size_pixels = load_vars_dict['image_size_pixels']
        pixel_size_arcmin = load_vars_dict['pixel_size_arcmin']
    
    y_map = _make_y_submap(profile, M200_SM, redshift_z, load_vars_dict,
                           image_size_pixels, pixel_size_arcmin,
                           alpha = alpha, gamma = gamma, R200_Mpc = R200_Mpc)

    return y_map

def get_r200_angsize_and_c200(M200_SM, redshift_z, load_vars_dict, angsize_density = None):
    '''
    Parameters:
    ----------
    M200_SM: float
        the mass contained within R200, in units of solar masses
    redshift_z: float
        redshift of the cluster (unitless)
    load_vars_dict: dict
        must contain 'cosmo' (a FlatLambaCDM instance describing the background cosmology),
        'sigma8' (float, around 0.8), and 'ns' (float, around 0.96)
    angsize_density: None or str
        density measure at which to calculate the angular size, if desired. If `None`, will not
        calculate an angular size. Otherwise, use a valid choice as specified in `colossus.halo.mass_adv`
        such as `500c`

    Returns:
    -------
    M200_SM: float
        the mass contained within R200, in units of solar masses
    R200_Mpc: float
        the radius of the cluster at 200 times the critical density of the universe in Mpc
    c200: float
        concentration parameter
    '''
    
    cosmo = load_vars_dict['cosmo']
    
    params = {'flat': True, 'H0': cosmo.H0.value, 'Om0': cosmo.Om0,
              'Ob0': cosmo.Ob0, 'sigma8': load_vars_dict['sigma8'], 'ns': load_vars_dict['ns']}
    cosmology.addCosmology('myCosmo', **params)
    cosmo_colossus = cosmology.setCosmology('myCosmo')
    
    M200_SM, R200_kpc, c200 = mass_adv.changeMassDefinitionCModel(M200_SM * cosmo.h,
                                                                  redshift_z, '200c', '200c', c_model = 'ishiyama21')
    M200_SM /= cosmo.h  # From M_solar/h to M_solar
    R200_Mpc = R200_kpc / cosmo.h / 1000  # From kpc/h to Mpc
    
    if angsize_density is not None:
        if angsize_density != '200c':
            _, Rd_kpc, _ = mass_adv.changeMassDefinitionCModel(M200_SM * cosmo.h,
                                                               redshift_z, '200c', angsize_density,
                                                               c_model = 'ishiyama21')
            angsize_arcmin = Rd_kpc / cosmo.h / 60 / cosmo_colossus.kpcPerArcsec(redshift_z)
        else:
            angsize_arcmin = R200_Mpc * 1000 / 60 / cosmo_colossus.kpcPerArcsec(redshift_z)
    else:
        angsize_arcmin = None
    return M200_SM, R200_Mpc, angsize_arcmin, c200 # now returns M200, R200, angsize in arcmin, c200
