import pytest
from simsz.make_sz_cluster import (
    P200_Battaglia2012, _param_Battaglia2012, Pth_Battaglia2012, epp_to_y, _make_y_submap, generate_y_submap,
    get_r200_and_c200)
import simsz.utils as utils
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

'''
Tests for the make_sz_cluster.py file, only testing functions used in simulation 
of cluster using Battaglia 2012 profile. 

Constants used in testing are similiar to those used in Battaglia 2012 paper
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
        M200 and R200 are the mass and radius of the cluster at 200 times the 
        critical density of the universe P200 is the thermal pressure profile of
        the shell defined by R200
        '''
        redshift_z = 1
        (cosmo,sigma8,ns) = get_mock_cosmology()
        M200 = 1e14
        P200_expected = 0.00137651 * (u.keV/u.cm**3.)
        P200_calculated = P200_Battaglia2012(M200, redshift_z, {'cosmo': cosmo, 'sigma8': 0.8, 'ns': 0.96})
        assert u.isclose(P200_calculated,P200_expected),f"Expected {P200_expected}, but got {P200_calculated}"
    
    def test_param_Battaglia2012(self):
        '''
        Test for the method param_Battaglia2012,
        which calculates independent params as defined in Battaglia 2012, 
        Equation 11. 
        These parameters are used to make the profile as described in Eq 10
        P0 is the normalization factor/amplitude,
        xc fits for the core-scale
        beta is a power law index
        '''
        redshift_z = 1
        M200 = 1e14 # in solar masses
        P0_expected = 10.702810552209492
        xc_expected = 0.8249152180764426
        beta_expected = 5.799849245346713
        assert u.isclose(
        _param_Battaglia2012(18.1, 0.154, -0.758, M200, redshift_z),
        P0_expected), "Incorrect param calculation: P0"
        assert u.isclose(
        _param_Battaglia2012(0.497,-0.00865,0.731,M200,redshift_z),
        xc_expected), "Incorrect param calculation: xc"
        assert u.isclose(
        _param_Battaglia2012(4.35,0.0393,0.415,M200,redshift_z)
        , beta_expected), "Incorrect param calculation: beta"

    def test_Pth_Battaglia2012(self):
        '''
        Test for the method Pth_Battaglia2012,
        which calculates Pth using the battaglia fit profile, Battaglia 2012, 
        Equation 10
        Pth is the thermal pressure profile normalized over P200
        P0 is the normalization factor/amplitude,
        xc fits for the core-scale
        beta is a power law index
        M200 and R200 are the mass and radius of the cluster at 200 times the 
        critical density of the universe
        P200 is the thermal pressure profile of the shell defined by R200
        '''
        radii=np.linspace(0.01,10,10000) #Generate a space of radii in arcmin
        (cosmo, sigma8, ns) = get_mock_cosmology()
        radii=utils.arcmin_to_Mpc(radii,0.5,cosmo)
        M200 = 1e14
        z = 1
        R200 = 0.8493839914731125
        x = radii/R200 #As defined in Battaglia 2012
        P0 = 10.702810552209492
        xc = 0.8249152180764426
        beta = 5.799849245346713
        Pth_expected = P0 * (x/xc)**(-0.3) * (1 + (x/xc))**(-beta)
        result = Pth_Battaglia2012(radii, M200, z, {'cosmo': cosmo, 'sigma8': 0.8, 'ns': 0.96}, 1.0, -0.3)
        assert np.allclose(result, Pth_expected),f"Expected {Pth_expected}, but got {result}"

    def test_Pe_to_y(self):
        '''
        Test for the method epp_to_y,
        which...
        '''
        (cosmo,sigma8,ns) = get_mock_cosmology()
        radii=np.linspace(0.01,10,10000) #Generate a space of radii in arcmin
        radii=utils.arcmin_to_Mpc(radii,0.5,cosmo)
        redshift_z = 1
        M200 = 1e14
        y = Pe_to_y(Pth_Battaglia2012, radii, M200, redshift_z, {'cosmo': cosmo, 'sigma8': 0.8, 'ns': 0.96}, 1.0, -0.3)
        assert np.max(y)==y[0]
    
    def test_make_y_submap(self):
        '''
        Test for the method _make_y_submap,
        which...
        '''
        (cosmo,sigma8,ns) = get_mock_cosmology()
        radii=np.linspace(0.01,10,10000) #Generate a space of radii in arcmin
        radii=utils.arcmin_to_Mpc(radii,0.5,cosmo)
        redshift_z = 1
        M200 = 1e14 #solar masses
        y = Pe_to_y(Pth_Battaglia2012, radii, M200, redshift_z, {'cosmo': cosmo, 'sigma8': 0.8, 'ns': 0.96}, 1.0, -0.3)
        y_map = _make_y_submap(Pth_Battaglia2012, M200, redshift_z, {'cosmo': cosmo, 'sigma8': 0.8, 'ns': 0.96}, 41,
                               0.5, 1.0, -0.3)
        fSZ_150GhZ = -0.9529784143018927
        dT_map = (y_map * cosmo.Tcmb0 * fSZ_150GhZ).to(u.uK).value
        y_expected = 1.1667524019195264e-05
        dT_expected = 30.29899851779931
        assert np.isclose(y_map.max(), y_expected),f"Expected {y_expected}, but got {y_map.max()}"
        assert np.isclose(abs(dT_map).max(), dT_expected),f"Expected {dT_expected}, but got {abs(dT_map).max()}"

