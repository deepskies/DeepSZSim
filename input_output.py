import yaml
from yaml.loader import SafeLoader


with open('config.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

    mass = data['mass']
    z = data['redshift']
    tele = data['telescope']
    freq = data['frequency']

    cosmo = data['COSMOLOGY']
    omega_m = cosmo['Omega_m0']
    omega_b = cosmo['Omega_b0']
    h = cosmo['cosmo_h']
    sigma8 = cosmo['sigma8']
    ns = cosmo['ns']

    telescope = data[tele]
    telescope_f = telescope[str(fre)+'GHz']
    beam_size = telescope_f['beam_size']
    noise_level = telescope_f['noise_level']
