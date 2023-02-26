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

def calc_scale_factor(z):
    a = 1. + z

    return a


def calc_rho_critical(a):
    rho_critical = (3 * H_0 ** 2)/(8 * np.pi * G) * (omega_m0 * a ** 3 + omega_d0) / m_sun * Mpc_to_m ** 3       #msolar / MPC^3

    return rho_critical
