import pytest
from make_sz_cluster import P200_Battaglia2012, param_Battaglia2012, Pth_Battaglia2012
import simsz.utils as utils
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

'''
Tests for the make_sz_cluster.py file, only testing functions used in simulation of cluster using 
Battaglia 2012 profile. 

Constants used in testing are similiar to those used in Battaglia 2012 paper
In order to run tests, use the command `PYTHONPATH=.. pytest` while in the 'tests' directory.
'''

def get_mock_cosmology():
    '''
    Generates a mock cosmology for testing purposes
    Returns a tuple of an Astropy FlatLambdaCDM cosmology,
    a sigma8 value, and a ns (the scalar spectral index)
    '''
    cosmo = FlatLambdaCDM(70, 0.25, Tcmb0=2.725, Ob0=0.043)
    sigma8 = 0.8
    ns = 0.96
    return (cosmo,sigma8,ns)

class TestSZCluster:
    def test_P200_Battaglia2012(self):
        '''
        Added testing for the method P200_Battaglia2012,
        which calculates the P200 param as defined in Battaglia 2012
        M200 and R200 are the mass and radius of the cluster at 200 times the critical density of the universe
        P200 is the thermal pressure profile of the shell defined by R200
        '''
        redshift_z = 0
        (cosmo,sigma8,ns) = get_mock_cosmology()
        M200 = 1.3e13
        R200 = 0.386
        P200_expected = 0.00014312182 * (u.keV/u.cm**3.)
        P200_calculated = P200_Battaglia2012(cosmo, redshift_z, M200, R200)
        assert u.isclose(P200_calculated,P200_expected), f"Expected {P200_expected}, but got {P200_calculated}"
    
    def test_param_Battaglia2012(self):
        '''
        Test for the method param_Battaglia2012,
        which calculates independent params as defined in Battaglia 2012, Equation 11 
        These parameters are used to make the profile as described in Eq 10
        P0 is the normalization factor/amplitude,
        xc fits for the core-scale
        beta is a power law index
        '''
        redshift_z = 0
        M200 = 1.3 * 10e13 # in solar masses
        P0_expected = 18.84628919814473
        xc_expected = 0.49587336181740654
        beta_expected = 4.395084514715711
        assert u.isclose(param_Battaglia2012(18.1,0.154,-0.758,M200,redshift_z), P0_expected), "Incorrect param calculation: P0"
        assert u.isclose(param_Battaglia2012(0.497,-0.00865,0.731,M200,redshift_z), xc_expected), "Incorrect param calculation: xc"
        assert u.isclose(param_Battaglia2012(4.35,0.0393,0.415,M200,redshift_z), beta_expected), "Incorrect param calculation: beta"

    def test_Pth_Battaglia2012(self):
        '''
        Test for the method Pth_Battaglia2012,
        which calculates Pth using the battaglia fit profile, Battaglia 2012, Equation 10
        Pth is the thermal pressure profile normalized over P200
        P0 is the normalization factor/amplitude,
        xc fits for the core-scale
        beta is a power law index
        M200 and R200 are the mass and radius of the cluster at 200 times the critical density of the universe
        P200 is the thermal pressure profile of the shell defined by R200
        '''
        radii=np.linspace(0.01,10,10000) #Generate a space of radii in arcmin
        (cosmo, sigma8, ns) = get_mock_cosmology()
        radii=utils.arcmin_to_Mpc(radii,0.5,cosmo)
        P0 = 18.84628919814473
        xc = 0.49587336181740654
        beta = 4.395084514715711
        R200 = 0.386
        x = radii/R200 #As defined in Battaglia 2012
        Pth_expected = P0 * (x/xc)**(-0.3) * (1 + (x/xc))**(-beta)
        result = Pth_Battaglia2012(radii,R200,-0.3,1.0,beta,xc,P0)
        assert np.allclose(result, Pth_expected), f"Expected {Pth_expected}, but got {result}"