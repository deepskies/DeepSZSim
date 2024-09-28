"""
creates a mass and redshift distribution of halos 
"""

import numpy as np
        
def flatdist_halo(zmin, zmax, m200min_SM, m200max_SM, size, seed=None):
    '''
    Creates a random uniform distribution of redshifts and masses for use in 
    creating simulations.
    
    Parameters:
    ----------
    zmin: float
        minimum value of the redshift distribution
    zmax: float
        maximum value of the redshift distribution
    m200min_SM: float
        minimum value of the mass distribution, in units of solar masses.
    m200max_SM: float
        maximum value of the mass distribution, in units of solar masses
    size: int
        size of the distribution
    seed: int
        seed for random number generation

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
    mdist=_rng.uniform(low=m200min_SM, high=m200max_SM, size=size)
    
    return zdist, mdist
