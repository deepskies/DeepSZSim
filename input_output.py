import yaml
from yaml.loader import SafeLoader


with open('inputdata.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

    cosmo = data['COSMOLOGY']
    omega_m = cosmo['Omega_m0']
    omega_b = cosmo['Omega_b0']
    h = cosmo['cosmo_h']
    sigma8 = cosmo['sigma8']
    ns = cosmo['ns']
    
    const = data['UNIVERSAL_CONSTANTS']
    planck_const = const['planck_const']
    boltzman_const = const['boltzman_const']
    G = const['G']
    m_sun = const['m_sun']
    Thomson_sec = const['Thomson_sec']
    m_electron = const['m_electron']
    c = const['c']
    Mpc_to_m = const['Mpc_to_m']
    kevcm_to_jm = const['kevcm_to_jm']
    j_to_kev = const['j_to_kev']
    t_cmb = const['t_cmb']
    
    imgs = data['IMAGES']
    image_size = imgs['image_size']
    pixel_scale = imgs['pixel_scale']

    tele = data['SURVEYS']
    telescope_f = tele['frequency'] #GHz
    beam_size = tele['beam_size']
    noise_level = tele['noise_level']
