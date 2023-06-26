import numpy as np
from colossus.cosmology import cosmology
from colossus.halo import mass_adv

class GenerateHalo():

    def __init__(self):
        self.data =[]
        
    def flatdist_halo(self,zmin,zmax,m500min,m500max,size):
        zdist=np.random.uniform(low=zmin, high=zmax, size=size)
        mdist=np.random.uniform(low=m500min, high=m500max, size=size)
        
        return zdist, mdist
        
    def vir_to_200_colossus(self,cosmo,sigma8,ns,Mvir,z):


         params = {'flat': True, 'H0': cosmo.H0.value, 'Om0': cosmo.Om0, 'Ob0': cosmo.Ob0, 'sigma8':sigma8, 'ns': ns}
         cosmology.addCosmology('myCosmo', **params)
         cosmo_colossus= cosmology.setCosmology('myCosmo')

         M200, R200, c200 = mass_adv.changeMassDefinitionCModel(Mvir/cosmo.h, z, 'vir', '200c', c_model = 'ishiyama21')

         M200 *= cosmo.h #From M_solar/h to M_solar
         R200 = R200*cosmo.h/1000 #From kpc/h to Mpc
         R200 = R200/cosmo.scale_factor(z) #From Mpc proper to Mpc comoving

         return (M200,R200,c200)