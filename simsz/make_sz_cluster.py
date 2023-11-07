import numpy as np
from simsz import utils, simtools, noise, load_vars
from colossus.cosmology import cosmology
from colossus.halo import mass_adv

from astropy import constants as c
from astropy import units as u
import os
import h5py
from datetime import datetime as dt
import shutil

keVcm_to_Jm = (1 * u.keV / (u.cm**3.)).to(u.J / (u.m**3.))
thermal_to_electron_pressure = 1 / 1.932  # from Battaglia 2012, assumes
# fully ionized medium

def _param_Battaglia2012(A0, alpha_m, alpha_z, M200_SM, redshift_z):
    '''
    Calculates independent params as using the formula from Battaglia 2012, Equation 11
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
    cosmo: FlatLambaCDM instance
        background cosmology for density calculation

    Returns:
    --------
    P200_kevcm3: Quantity instance
        the thermal pressure of the shell defined by R200 in units
        of keV/cm**3
    '''
    cosmo = load_vars_dict['cosmo']

    if R200_Mpc is None:
        R200_Mpc = get_r200_and_c200(M200_SM, redshift_z, load_vars_dict)[1]
    
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
        the radius for the pressure to be calculated at, in units of Mpc
    R200_Mpc: float
        the radius of the cluster at 200 times the critical density of the
        universe, in units of Mpc
    alpha: float
        fixed parameter defined by Battaglia 2012
    beta: float
        power law index
    gamma: float
        fixed paremeter defined by Battaglia 2012
    xc: float
        fits for the core-scale

    Returns:
    --------
    Pth: float or np.ndarray(float)
        the thermal pressure profile normalized by P200 (which itself has units of
        keV/cm**3)
    '''
    
    x = radius_mpc / R200_Mpc
    
    Pth = P0 * (x / xc)**gamma * (1 + (x / xc)**alpha)**(-beta)
    
    return (Pth)

def Pth_Battaglia2012(radius_mpc, M200_SM, redshift_z, load_vars_dict, alpha = 1.0, gamma = -0.3, R200_Mpc = None):
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
    cosmo: FlatLambaCDM instance
        background cosmology for density calculation
    alpha: float
        variable fixed by Battaglia et al 2012 to 1.0
    gamma: float
        variable fixed by Battaglia et al 2012 to -0.3

    Returns:
    --------
    Pth: float or np.ndarray(float)
        the thermal pressure profile normalized by P200 (which itself has units of
        keV/cm**3)
    '''
    
    if R200_Mpc is None:
        R200_Mpc = get_r200_and_c200(M200_SM, redshift_z, load_vars_dict)[1]
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
    cosmo: FlatLambaCDM instance
        background cosmology for density calculation
    alpha: float
        variable fixed by Battaglia et al 2012 to 1.0
    gamma: float
        variable fixed by Battaglia et al 2012 to -0.3

    Return:
    -------
    y_pro: array
        Compton-y profile corresponding to the radii
    '''
    radii_mpc = radii_mpc * u.Mpc
    pressure_integ = np.empty(radii_mpc.size)
    P200_kevcm3 = P200_Battaglia2012(M200_SM, redshift_z, load_vars_dict)
    if profile == "Battaglia2012":
        profile = Pth_Battaglia2012
    
    for i, radius in enumerate(radii_mpc):
        # Multiply profile by P200 specifically for Battaglia 2012 profile,
        # since it returns Pth/P200 instead of Pth
        rv = radius.value
        l_mpc = np.linspace(0, np.sqrt(radii_mpc.value.max()**2. - rv**2.)+1.e-5, 1000)  # Get line of sight
        # axis
        th_pressure = profile(np.sqrt(l_mpc**2 + rv**2), M200_SM, redshift_z, load_vars_dict, alpha = alpha,
                              gamma = gamma, R200_Mpc = R200_Mpc)
        th_pressure = th_pressure * P200_kevcm3.value  # pressure as a
        #                                               function of l
        th_pressure = th_pressure * keVcm_to_Jm.value  # Use multiplication
        #                           by a precaluated factor for efficiency
        pressure = th_pressure * thermal_to_electron_pressure
        integral = np.trapz(pressure, l_mpc*(1 * u.Mpc).to(u.m).value) * 2  # integrate over pressure in
        # J/m^3 to get J/m^2, multiply by factor of 2 to get from -R200 to
        # R200 (assuming spherical symmetry)
        pressure_integ[i] = integral
    y_pro = pressure_integ * c.sigma_T.value / (c.m_e.value * c.c.value**2)
    return y_pro


def _make_y_submap(profile, M200_SM, redshift_z, load_vars_dict, image_size, pix_size_arcmin, alpha = 1.0, gamma = -0.3, R200_Mpc = None):
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
    image_size: float
        size of final submap
    pix_size_arcmin: float
        size of each pixel in arcmin

    Return:
    -------
    y_map: array
        Compton-y submap with shape (image_size, image_size)
    '''
    
    X = np.linspace(-image_size * pix_size_arcmin / 2,
                    image_size * pix_size_arcmin / 2, image_size)
    X = utils.arcmin_to_Mpc(X, redshift_z, load_vars_dict['cosmo'])
    # Solves issues of div by 0
    X[(X <= pix_size_arcmin / 10) & (X >= -pix_size_arcmin / 10)] = pix_size_arcmin / 10
    
    y_map = np.empty((X.size, X.size))
    
    R = np.sqrt(X[:,None]**2 + X[None,:]**2).flatten()
    cy = Pe_to_y(profile, R, M200_SM, redshift_z, load_vars_dict, alpha = alpha, gamma = gamma, R200_Mpc = R200_Mpc)  #
    # evaluate compton-y for each
    # neccesary radius
    
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            y_map[i][j] = cy[np.where(np.isclose(R, np.sqrt(x**2 + y**2),
                                                 atol=1.e-10, rtol=1.e-10))[0]][0]
    # assign the correct compton-y to the radius
    
    return y_map


def generate_y_submap(M200_SM, redshift_z, profile = "Battaglia2012", cosmo = None,
                      image_size = None, pix_size_arcmin = None, load_vars_dict = None, alpha = 1.0, gamma = -0.3,
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
        Name of profile, currently only supports "Battaglia2012"
    cosmo: FlatLambaCDM instance
        background cosmology for density calculation
    image_size: float
        num pixels to each side of center; end shape of submap will be 
        (2*image_size +1, 2*image_size +1)
    pix_size_arcmin: float
        size of each pixel in arcmin
    load_vars_dict: dict
        result of running the load_vars() function, which includes a dictionary of cosmological and experimental
        parameters

    Return:
    ------
    y_map: array
        Compton-y submap with dimension (2*image_size +1 , 2*image_size +1)
    '''
    if profile != "Battaglia2012":
        print("only implementing Battaglia2012")
        return None
    
    if load_vars_dict is not None:
        image_size = load_vars_dict['image_size_arcmin']
        pix_size_arcmin = load_vars_dict['pix_size_arcmin']
    
    y_map = _make_y_submap(profile, M200_SM, redshift_z, load_vars_dict,
                           image_size, pix_size_arcmin,
                           alpha = alpha, gamma = gamma, R200_Mpc = R200_Mpc)
    
    return y_map


def simulate_T_submaps(M200_dist, z_dist, id_dist = None, profile = "Battaglia2012",
                    savedir = os.path.join(os.getcwd(), 'outfiles'), saverun = False,
                    R200_dist = None, add_cmb = True,
                    load_vars_yaml = os.path.join(os.path.dirname(__file__), 'Settings',
                                                 'inputdata.yaml')):
    """
    Simulates a dT map for a cluster using each M200, z pair, using the density
    profile from Battaglia 2012
    Uses params from Settings/inputdata.yml

    Parameters:
    ----------
    M200_dist: float or array-like of float
        the mass contained within R200 in solar masses (same length as z_dist)
    z_dist: float or array-like of float
        the redshift of the cluster (unitless) (same length as M200_dist)
    id_dist: float or array-like of float, optional
        id of the sim or cluster (same length as M200_dist), 
        generated if not given
    savedir : str, default CWD/outfiles
            directory into which results will be saved
    R200_dist: float or array-like of float, optional
        the radius of the cluster at 200 times the critical density of the 
        universe in Mpc (same length as M200_dist), calculated via colossus 
        if not given
    add_cmb: bool
        To add background cmb or not, defualt True
    settings_yaml : str, default CWD/simsz/Settings/inputdata.yaml
            path to yaml file with params

    Return:
    ------
    clusters: array of dicts
        Each dict contains the full information of each sim/cluster. 
        Dict has attributes:
        M200, R200, redshift_z, y_central, ID, cmb_map, noise_map
        final_map

    """
    # Make a dictionary and cosmology from the .yaml
    d = load_vars.load_vars(load_vars_yaml) # eventually, make compatible with all the load_vars kwargs
    
    M200_dist = np.asarray(M200_dist)
    z_dist = np.asarray(z_dist)
    if add_cmb:
        # To make sure to only calculate this once if its a dist
        ps = simtools.get_cls(ns = d['ns'], cosmo = d['cosmo'])
    
    if not os.path.exists(savedir):
        print(f"making local directory `{savedir}`")
        os.mkdir(savedir)
    
    # Generate a run_id based on time of running and freq
    rand_num = np.random.randint(10**6)
    run_id = dt.now().strftime('%y%m%d%H%M%S%f_') + str(d['survey_freq']) + '_' + str(
        rand_num).zfill(6)
    
    if saverun:
        f = h5py.File(os.path.join(savedir, f'sz_sim_{run_id}.h5'), 'a')
    
    clusters = []
    
    for index, M200 in enumerate(M200_dist):
        z = z_dist[index]
        if R200_dist is None:
            (M200, R200, _c200) = get_r200_and_c200(M200, z, d)
        else:
            R200 = R200_dist[index]
        
        y_map = generate_y_submap(M200, z, profile = profile, load_vars_dict = d, R200_Mpc = R200)
        # get f_SZ for observation frequency
        fSZ = simtools.f_sz(d['survey_freq'], d['cosmo'].Tcmb0)
        dT_map = (y_map * d['cosmo'].Tcmb0 * fSZ).to(u.uK)
        
        cluster = {'M200': M200, 'R200': R200, 'redshift_z': z,
                   'y_central': y_map[d['image_size_arcmin'] // 2][d['image_size_arcmin'] // 2]}
        
        if id_dist is not None:
            cluster['ID'] = id_dist[index]
        else:
            # Generate a simID
            rand_num = np.random.randint(10**6)
            cluster['ID'] = str(M200)[:5] + str(z * 100)[:2] + str(rand_num).zfill(6)
        
        if add_cmb:
            conv_map, cmb_map = simtools.add_cmb_map_and_convolve(dT_map,
                                                                  ps,
                                                                  d['pix_size_arcmin'],
                                                                  d['beam_size_arcmin'])
            cluster['CMB_map'] = cmb_map
        
        else:
            conv_map = simtools.convolve_map_with_gaussian_beam(
                d['pix_size_arcmin'], d['beam_size_fwhp_arcmin'], dT_map)
        
        if not d['noise_level'] == 0:
            noise_map = noise.generate_noise_map(d['image_size_arcmin'],
                                                 d['noise_level'], d['pix_size_arcmin'])
            final_map = conv_map + noise_map
            
            cluster['noise_map'] = noise_map
            cluster['final_map'] = final_map
        
        clusters.append(cluster)
        if saverun:
            utils.save_sim_to_h5(f, f"sim_{cluster['ID']}", cluster)
    
    if saverun:
        f.close()
        shutil.copyfile(os.path.join(os.path.dirname(__file__), "Settings", "inputdata.yaml"),
                        os.path.join(savedir, f'params_{run_id}.yaml'))
    
    return clusters


def get_r200_and_c200(M200_SM, redshift_z, load_vars_dict):
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
    
    M200_SM, R200_Mpc, c200 = mass_adv.changeMassDefinitionCModel(M200_SM / cosmo.h,
                                                                  redshift_z, '200c', '200c', c_model = 'ishiyama21')
    M200_SM *= cosmo.h  # From M_solar/h to M_solar
    R200_Mpc = R200_Mpc * cosmo.h / 1000  # From kpc/h to Mpc
    # From Mpc proper to Mpc comoving
    R200_Mpc = R200_Mpc / cosmo.scale_factor(redshift_z)
    return M200_SM, R200_Mpc, c200
