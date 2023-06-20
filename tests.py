import pytest
import make_sz_cluster
import get_dm_halo
import utils
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

# Overall, constants close to ones used in Battaglia 2012
# calculated expected values based off of paper, however unit problems maye still exist

# Largest problem with the tests is that Ive used the same methods as the code to
# calculate the constants/expected anwsers, so if there is a major mistake the test will replicate that mistake
# These will be more useful in the future once we've verified these caluclations are correct

# Makes a mock cosmology 
def get_mock_cosmology():
    cosmo = FlatLambdaCDM(70, 0.25, Tcmb0=2.725, Ob0=0.043)
    sigma8 = 0.8
    ns = 0.96
    return (cosmo,sigma8,ns)

class TestSZCluster:
    def test_P200_Battaglia2012(self):
        cluster = make_sz_cluster.GenerateCluster()
        (cosmo, sigma8, ns) = get_mock_cosmology()
        z = 0
        M200 = 1.3e13
        R200 = 0.386
        P200_expected = 0.00014312182 * (u.keV/u.cm**3.)
        assert u.isclose(cluster.P200_Battaglia2012(cosmo, z, M200, R200),P200_expected), "Incorrect P200 value"
    
    def test_param_Battaglia2012(self):
        cluster = make_sz_cluster.GenerateCluster()
        z = 0
        M200 = 1.3 * 10e13 # in solar masses
        
        P0_expected = 18.84628919814473
        xc_expected = 0.49587336181740654
        beta_expected = 4.395084514715711
        assert u.isclose(cluster.param_Battaglia2012(18.1,0.154,-0.758,M200,z), P0_expected), "Incorrect param calculation: P0"
        assert u.isclose(cluster.param_Battaglia2012(0.497,-0.00865,0.731,M200,z), xc_expected), "Incorrect param calculation: xc"
        assert u.isclose(cluster.param_Battaglia2012(4.35,0.0393,0.415,M200,z), beta_expected), "Incorrect param calculation: beta"

    def test_Pth_Battaglia2012(self):
        cluster = make_sz_cluster.GenerateCluster()
        z = 0
        r=np.linspace(0.01,10,10000) #Generate a space of radii in arcmin
        (cosmo, sigma8, ns) = get_mock_cosmology()
        r=utils.arcmin_to_Mpc(r,0.5,cosmo)
        P0 = 18.84628919814473
        xc = 0.49587336181740654
        beta = 4.395084514715711
        M200 = 1.3e13
        R200 = 0.386
        P200 = 0.00014312182 * (u.keV/u.cm**3.)
        x = r/R200
        Pth_expected = P0 * (x/xc)**(-0.3) * (1 + (x/xc))**(-beta)
        assert np.allclose(cluster.Pth_Battaglia2012(cosmo,r,z,R200,-0.3,1.0,beta,xc,P0,P200,M200), Pth_expected), "Incorrect Pth calculations"
    
    #I need to actually understand what is going on here in order to say what the result should be

class TestDMHalo:
    def test_vir_to_200(self):
        halo = get_dm_halo.GenerateHalo()
        (cosmo, sigma8, ns) = get_mock_cosmology()
        #Figure out what this is actually doing, then come up with mock/test cases
        M200_expected = 12979452205744.412
        R200_expected = 0.38689299677471767
        c200_expected = 4.046612792450555
        (M200,R200,c200) = halo.vir_to_200_colossus(cosmo,sigma8,ns,Mvir=1.5e13,z=0.5)
        assert np.isclose(M200_expected, M200), "Incorrect M200"
        assert np.isclose(R200_expected, R200), "Incorrect R200"
        assert np.isclose(c200_expected, c200), "Incorrect c200"
