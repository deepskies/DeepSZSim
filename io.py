import yaml
from yaml.loader import SafeLoader


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