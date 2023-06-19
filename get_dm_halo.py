import numpy as np

class GenerateHalo():

    def __init__(self):
        self.data =[]
        
    def flatdist_halo(self,zmin,zmax,m500min,m500max,size):
        zdist=np.random.uniform(low=zmin, high=zmax, size=size)
        mdist=np.random.uniform(low=m500min, high=m500max, size=size)
        
        return zdist, mdist
        
