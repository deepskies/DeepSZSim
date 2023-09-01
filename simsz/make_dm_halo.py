import numpy as np
from colossus.cosmology import cosmology
from colossus.halo import mass_adv
        
def flatdist_halo(zmin,zmax,m500min,m500max,size):
    zdist=np.random.uniform(low=zmin, high=zmax, size=size)
    mdist=np.random.uniform(low=m500min, high=m500max, size=size)
    
    return zdist, mdist