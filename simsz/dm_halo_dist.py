import numpy as np
from colossus.cosmology import cosmology
from colossus.halo import mass_adv
        
def flatdist_halo(zmin, zmax, m500min_SM, m500max_SM, size, seed=None):
    '''
    Creates a random uniform distribution of redshifts and masses for use in 
    creating simulations.
    
    Parameters:
    ----------
    zmin: float
        minimum value of the redshift distribution
    zmax: float
        maximum value of the redshift distribution
    m500min_SM: float
        minimum value of the mass distribution, in units of solar masses.
    m500max_SM: float
        maximum value of the mass distribution, in units of solar masses
    size: int
        size of the distribution

    Returns:
    -------
    zdist: float array
        distribution of random uniform redshifts starting at `zmin` ending at 
        `zmax` with size `size`
    mdist: float array
        distribution of random uniform redshifts starting at `m500min_SM` ending
        at `m500max_SM` with size `size`
    '''
    _rng = np.random.default_rng(seed=seed)

    zdist=_rng.uniform(low=zmin, high=zmax, size=size)
    mdist=_rng.uniform(low=m500min_SM, high=m500max_SM, size=size)
    
    return zdist, mdist