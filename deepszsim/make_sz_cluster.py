"""
pressure profile, Compton-y, R200, C200 and temperature submap generating functions based on halo redshift and mass information
"""

import numpy as np
from deepszsim import utils, simtools, noise, load_vars, dm_halo_dist
from colossus.cosmology import cosmology
from colossus.halo import mass_adv
import abel
from tqdm import tqdm

from astropy import constants as c
from astropy import units as u
import os
import h5py
from datetime import datetime as dt
import shutil

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
        R200_Mpc = get_r200_and_c200(M200_SM, redshift_z, load_vars_dict)[1]
    P0 = _P0_Battaglia2012(M200_SM, redshift_z)
    xc = _xc_Battaglia2012(M200_SM, redshift_z)
    beta = _beta_Battaglia2012(M200_SM, redshift_z)
    
    return _Pth_Battaglia2012(P0, radius_mpc, R200_Mpc, alpha, beta, gamma, xc)


def Pe_to_y(profile, radii_mpc, M200_SM, redshift_z, load_vars_dict, alpha = 1.0, gamma = -0.3, R200_Mpc = None,
            Rmaxy = None):
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
        R200_Mpc = get_r200_and_c200(M200_SM, redshift_z, load_vars_dict)[1]
    radii_mpc = (radii_mpc * u.Mpc).value
    if Rmaxy is None:
        rmax = radii_mpc.max()
    elif '200' in Rmaxy:
        rmax = R200_Mpc
    else:
        print('please specify a valid `Rmaxy`')
        return None
    if profile != "Battaglia2012":
        print("only implementing `Battaglia2012` for profile")
    profile = Pth_Battaglia2012
    pressure_integ = np.empty_like(radii_mpc)
    P200_kevcm3 = P200_Battaglia2012(M200_SM, redshift_z, load_vars_dict, R200_Mpc = R200_Mpc).value
    
    # integral = np.trapz(np.array([profile(np.sqrt(np.linspace(0, np.sqrt(radii_mpc.max()**2. - rv**2.)+1.,
    #                                                               1000)**2 +
    #                                      rv**2), M200_SM, redshift_z, load_vars_dict = None, alpha = alpha,
    #                       gamma = gamma, R200_Mpc = r200) for rv in radii_mpc]), np.array([np.linspace(0,
    #                                                                                             np.sqrt(radii_mpc.max(
    # )**2. - rv**2.)+1., 1000) for rv in radii_mpc]))
    # y_pro = integral * P200_kevcm3 * keVcm3_to_Jm3 * Thomson_scale * \
    #         thermal_to_electron_pressure * 2*Mpc_to_m
    # return y_pro
    for i, radius in enumerate(radii_mpc):
        # Multiply profile by P200 specifically for Battaglia 2012 profile,
        # since it returns Pth/P200 instead of Pth
        rv = radius
        if (rmax == R200_Mpc) and (rv >= R200_Mpc):
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
                   gamma = -0.3, R200_Mpc = None, Rmaxy = None):
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
    
    cy = Pe_to_y(profile, R, M200_SM, redshift_z, load_vars_dict, alpha = alpha, gamma = gamma, R200_Mpc = R200_Mpc,
                 Rmaxy = Rmaxy)  #
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


def generate_y_submap(M200_SM, redshift_z, profile = "Battaglia2012", method = "integrate",
                      image_size_pixels = None, pixel_size_arcmin = None, load_vars_dict = None, alpha = 1.0, gamma = -0.3,
                      R200_Mpc = None, Rmaxy = None):
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
    method: str
        procedure by which we obtain the y map from a pressure profile. Two options at this time are
        "integrate" (using np.trapz), which is precise and inherently spherically symmetric, but slow; or
        "abel" (using pyabel), which suffers distortions along the y-axis but is fast. We recommend "integrate" for
        producing single maps and "abel" for producing large quantities of maps (eg for `sbi`).
    image_size_pixels: float
        num pixels to each side of center; end shape of submap will be 
        (image_size_pixels, image_size_pixels)
    pixel_size_arcmin: float
        size of each pixel in arcmin
    load_vars_dict: dict
        result of running the load_vars() function, which includes a dictionary of cosmological and experimental
        parameters
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
    dim = (image_size_pixels // 2) * pixel_size_arcmin
    
    if "abel" in method:
        X = np.linspace(- dim, dim, image_size_pixels)
        pixlocs = utils.arcmin_to_Mpc(np.sqrt(X[:, None]**2 + X[None, :]**2), redshift_z, load_vars_dict['cosmo'])
        mindist = utils.arcmin_to_Mpc(pixel_size_arcmin * 0.1, redshift_z, load_vars_dict['cosmo'])
        pixlocs = np.maximum(mindist, pixlocs)
        if R200_Mpc is None:
            r200 = get_r200_and_c200(M200_SM, redshift_z, load_vars_dict)[1]
        else:
            r200 = R200_Mpc
        if (Rmaxy is not None) and ('200' in Rmaxy):
            pixlocs = np.minimum(r200, pixlocs)
        paDa = abel.transform.Transform(Pth_Battaglia2012(pixlocs, M200_SM, redshift_z, R200_Mpc = r200),
                                        direction = 'forward', method = 'daun',
                                        symmetry_axis = (0,1), transform_options = {'degree':2}
                                        ).transform
        paDa_sym = (paDa + paDa.T)/2
        y_map = paDa_sym * keVcm3_to_Jm3 * Mpc_to_m * thermal_to_electron_pressure * P200_Battaglia2012(
            M200_SM, redshift_z, load_vars_dict, R200_Mpc = r200).value * Thomson_scale
    else:
        if not ("integ" in method):
            print("only valid method choices are 'abel' or 'integrate'. Defaulting to 'integrate'")
        y_map = _make_y_submap(profile, M200_SM, redshift_z, load_vars_dict,
                               image_size_pixels, pixel_size_arcmin,
                               alpha = alpha, gamma = gamma, R200_Mpc = R200_Mpc, Rmaxy = Rmaxy)

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
    profile: str
        Name of profile, currently only supports "Battaglia2012"
    savedir : str, default CWD/outfiles
        directory into which results will be saved
    saverun: bool
        whether or not to save runs
    R200_dist: float or array-like of float, optional
        the radius of the cluster at 200 times the critical density of the 
        universe in Mpc (same length as M200_dist), calculated via colossus 
        if not given
    add_cmb: bool
        To add background cmb or not, defualt True
    load_vars_yaml : str, default CWD/deepszsim/Settings/inputdata.yaml
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
                   'y_central': y_map[d['image_size_pixels'] // 2][d['image_size_pixels'] // 2]}
        
        if id_dist is not None:
            cluster['ID'] = id_dist[index]
        else:
            # Generate a simID
            rand_num = np.random.randint(10**6)
            cluster['ID'] = str(M200)[:5] + str(z * 100)[:2] + str(rand_num).zfill(6)
        
        if add_cmb:
            conv_map, cmb_map = simtools.add_cmb_map_and_convolve(dT_map,
                                                                  ps,
                                                                  d['pixel_size_arcmin'],
                                                                  d['beam_size_arcmin'])
            cluster['CMB_map'] = cmb_map
        
        else:
            conv_map = simtools.convolve_map_with_gaussian_beam(
                d['pixel_size_arcmin'], d['beam_size_fwhp_arcmin'], dT_map)
        
        if not d['noise_level'] == 0:
            noise_map = noise.generate_noise_map(d['image_size_pixels'],
                                                 d['noise_level'], d['pixel_size_arcmin'])
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


class simulate_clusters:
    def __init__(self, M200 = None, redshift_z = None, num_halos = None, halo_params_dict = None,
                 R200_Mpc = None, Rmaxy = None, profile = "Battaglia2012",
                 image_size_pixels = None, image_size_arcmin = None, pixel_size_arcmin = None,
                 alpha = 1.0, gamma = -0.3,
                 load_vars_yaml = os.path.join(os.path.dirname(__file__), 'Settings', 'inputdata.yaml'),
                 seed = None, tqverb = False
                 ):
        """
        Parameters
        ----------
        M200_dist: float or array-like of float
            the mass contained within R200 in solar masses (same length as z_dist)
        z_dist: float or array-like of float
            the redshift of the cluster (unitless) (same length as M200_dist)
        num_halos: None or int
            number of halos to simulate if none supplied
        halo_params_dict: None or dict
            parameters from which to sample halos if num_halos specified,
            must contain zmin, zmax, m200min_SM, m200max_SM
        R200_Mpc: None or float or np.ndarray(float)
            if None, will calculate the R200 values corresponding to a given set of
            M200 and redshift_z values for the specified cosmology
        profile: str
            Name of profile, currently only supports "Battaglia2012"
        image_size_pixels: None or int
            image size in pixels (should be odd)
        image_size_arcmin: None or float
            image size in arcmin
        pixel_size_arcmin: None or float
            pixel size in arcmin
        alpha: float
            fixed to equal 1.0 in Battaglia 2012
        gamma: float
            fixed to equal -0.3 in Battaglia 2012
        load_vars_yaml: str
            path to yaml file with params
        seed: None or int
            random seed value to sample with
        """
        
        if (M200 is not None) and (redshift_z is not None):
            self.M200, self.redshift_z = M200, redshift_z
        else:
            if (num_halos is None):
                print("must specify `M200` AND `redshift_z` simultaneously,",
                      "OR a number of halos to generate with `num_halos`"
                      "along with the arguments for `deepszsim.dm_halo_dist.flatdist_halo` via `halo_params_dict`.",
                      "Defaulting to making 100 halos in 0.1<z<1.1, 1e14<M<1e15")
                num_halos = 100
            if (halo_params_dict is None):
                print(f"making {num_halos} clusters uniformly sampled from 0.1<z<1.1, 1e13<M200<1e14")
                halo_params_dict = {'zmin': 0.1, 'zmax': 1.1, 'm200min_SM': 1e13, 'm200max_SM': 1e14}
            self.redshift_z, self.M200 = dm_halo_dist.flatdist_halo(halo_params_dict['zmin'],
                                                                    halo_params_dict['zmax'],
                                                                    halo_params_dict['m200min_SM'],
                                                                    halo_params_dict['m200max_SM'],
                                                                    int(num_halos), seed = seed)
        
        try:
            self._size = len(self.M200)
        except TypeError:
            self.M200, self.redshift_z = np.array([self.M200]), np.array([self.redshift_z])
            self._size = 1
        self.clusters = {}
        
        if profile != "Battaglia2012":
            print("only `Battaglia2012` is implemented, using that for now")
        self.profile = "Battaglia2012"
        
        self.vars = load_vars.load_vars(load_vars_yaml)
        self.image_size_pixels = self.vars['image_size_pixels'] if (image_size_pixels is None) else image_size_pixels
        self.image_size_arcmin = self.vars['image_size_arcmin'] if (image_size_arcmin is None) else image_size_arcmin
        self.pixel_size_arcmin = self.vars['pixel_size_arcmin'] if (pixel_size_arcmin is None) else pixel_size_arcmin
        self.beam_size_arcmin = self.vars['beam_size_arcmin']
        self.cosmo = self.vars['cosmo']
        self.tqverb = tqverb
        
        self.alpha, self.gamma = alpha, gamma
        self.seed, self._rng = seed, np.random.default_rng(seed)
        
        if R200_Mpc is not None:
            self.R200_Mpc = R200_Mpc
        else:
            self.R200_Mpc = np.array(
                [get_r200_and_c200(self.M200[i], self.redshift_z[i], self.vars)[1]
                 for i in range(self._size)])
        
        self.Rmaxy = Rmaxy
        
        self.id_list = [
            str(self.M200[i])[:5] + str(self.redshift_z[i] * 100)[:2] + str(self._rng.integers(10**6)).zfill(6)
            for i in range(self._size)]
        self.clusters.update(zip(self.id_list, [{"params": {'M200': self.M200[i], 'redshift_z': self.redshift_z[i],
                                                            'R200': self.R200_Mpc[i], 'image_size_pixels' : self.image_size_pixels}} for
                                                i in range(
            self._size)]))
    
    def get_y_maps(self):
        """
        
        Returns
        -------
        np.ndarray(float)
            self._size many maps of the Compton `y` value, each of which is image_size_pixels x image_size_pixels in size
        """
        try:
            return self.y_maps
        except AttributeError:
            if self.tqverb: print("making `y` maps")
            self.y_maps = np.array([generate_y_submap(self.M200[i],
                                                      self.redshift_z[i],
                                                      R200_Mpc = self.R200_Mpc[i],
                                                      Rmaxy = self.Rmaxy,
                                                      load_vars_dict = self.vars)
                                    for i in tqdm(range(self._size), disable = (not self.tqverb))])
            return self.y_maps
    
    def get_dT_maps(self):
        """
        
        Returns
        -------
        np.ndarray(float)
            self._size many maps of the dT values in units of uK, each of which is image_size_pixels x
            image_size_pixels in size
        """
        try:
            return self.dT_maps
        except AttributeError:
            ym = self.get_y_maps()
            fSZ = simtools.f_sz(self.vars['survey_freq'], self.cosmo.Tcmb0)
            self.dT_maps = (ym * fSZ * self.cosmo.Tcmb0).to(u.uK).value
            return self.dT_maps
    
    def get_T_maps(self, add_CMB = True, returnval = False):
        """
        Parameters
        ----------
        add_CMB: bool
            whether or not to include the CMB contribution to the final map
        
        Returns
        -------
        np.ndarray(float)
            self._size many maps of the sky in units of uK, each of which is image_size_pixels x image_size_pixels in
            size
        """
        dT_maps = self.get_dT_maps()
        if add_CMB: self.ps = simtools.get_cls(ns = self.vars['ns'], cosmo = self.vars['cosmo'])
        if self.tqverb: print("making convolved T maps"+(" with CMB" if add_CMB else ""))
        for i in tqdm(range(self._size), disable = (not self.tqverb)):
            self.clusters[self.id_list[i]]['params']['dT_central'] = dT_maps[i][self.image_size_pixels // 2][
                self.image_size_pixels // 2]
            self.clusters[self.id_list[i]]['maps'] = {}
            if add_CMB:
                conv_map, cmb_map = simtools.add_cmb_map_and_convolve(dT_maps[i], self.ps,
                                                                            self.pixel_size_arcmin,
                                                                            self.beam_size_arcmin)
                self.clusters[self.id_list[i]]['maps']['CMB_map'] = cmb_map
            else:
                conv_map = simtools.convolve_map_with_gaussian_beam(
                    self.pixel_size_arcmin, self.beam_size_arcmin, dT_map)
            if not self.vars['noise_level'] == 0:
                noise_map = noise.generate_noise_map(self.image_size_pixels, self.vars['noise_level'],
                                                     self.pixel_size_arcmin)
            else:
                noise_map = np.zeros_like(conv_map)
            final_map = conv_map + noise_map
            self.clusters[self.id_list[i]]['maps']['conv_map'] = conv_map
            self.clusters[self.id_list[i]]['maps']['noise_map'] = noise_map
            self.clusters[self.id_list[i]]['maps']['final_map'] = final_map
        
        if returnval:
            return self.clusters
    
    def ith_T_map(self, i, add_CMB = True):
        """
        Parameters
        ----------
        i: int
            the map you want to return
        add_CMB: bool
            whether or not to include the CMB contribution to the final map
        
        Returns
        -------
        np.ndarray(float)
            the ith map of the sky in units of uK, which is image_size_pixels x image_size_pixels in size
        """
        try:
            return self.clusters[self.id_list[i]]['maps']['final_map']
        except KeyError:
            self.get_T_maps(add_CMB = add_CMB)
            return self.clusters[self.id_list[i]]['maps']['final_map']
    
    def save_map(self, i = None, nest_h5 = True, nest_name = None,
                 savedir = os.path.join(os.path.dirname(__file__), "outfiles")):
        """
        Parameters
        ----------
        i: None or int
            the map you want to save, if you only want to save a single map
        nest_h5: bool
            whether or not to nest the clusters into a single h5 file, assuming that you are saving all of the
            clusters that you have calculated
        nest_name: None or str
            a name for the overall file if you are nesting them (otherwise, it will name it with the number of
            clusters plus the date plus a random string)
        savedir: str
            the path to the directory you want to save into
        """
        self.savedir = savedir
        if not os.path.exists(self.savedir):
            print(f"making local directory `{self.savedir}`")
            os.mkdir(self.savedir)
        try:
            self.clusters[self.id_list[0]]['maps']
        except KeyError:
            self.get_T_maps()
        if i is not None:
            with h5py.File(os.path.join(self.savedir, self.id_list[i] + ".h5"), 'w') as f:
                for k, v in self.clusters[self.id_list[i]]['params'].items():
                    f.create_dataset('params/' + k, data = float(v))
                for k, v in self.clusters[self.id_list[i]]['maps'].items():
                    f.create_dataset('maps/' + k, data = v)
        elif nest_h5:
            file_name = str(self._size) + "clusters_" + dt.strftime(dt.now(),
                                                                    '%y%m%d%H%M%S%f') if nest_name is None else nest_name
            with h5py.File(os.path.join(self.savedir, file_name + ".h5"), 'w') as f:
                for j in range(self._size):
                    for k, v in self.clusters[self.id_list[j]]['params'].items():
                        f.create_dataset(self.id_list[j] + '/params/' + k, data = float(v))
                    for k, v in self.clusters[self.id_list[j]]['maps'].items():
                        f.create_dataset(self.id_list[j] + '/maps/' + k, data = v)
        else:
            for i in range(self._size):
                with h5py.File(os.path.join(self.savedir, self.id_list[i] + ".h5"), 'w') as f:
                    for k, v in self.clusters[self.id_list[i]]['params'].items():
                        f.create_dataset('params/' + k, data = float(v))
                    for k, v in self.clusters[self.id_list[i]]['maps'].items():
                        f.create_dataset('maps/' + k, data = v)

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
