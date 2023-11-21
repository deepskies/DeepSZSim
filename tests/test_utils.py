import numpy as np
import os
import pytest
from simsz.utils import arcmin_to_Mpc, Mpc_to_arcmin
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM


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

class TestUtils():
    def test_arcmin_to_Mpc(self):
        r = 1
        z = 0.5
        (cosmo,sigma8,ns) = get_mock_cosmology()
        value_expected = 0.559219683539306
        value_calculated = arcmin_to_Mpc(r, z, cosmo)
        assert u.isclose(value_calculated, value_expected),f"Expected {value_expected}, but got {value_calculated}"

    def test_Mpc_to_arcmin(self):
        r = 1
        z = 0.5
        (cosmo,sigma8,ns) = get_mock_cosmology()
        value_expected = 1.7882060117608733
        value_calculated = Mpc_to_arcmin(r, z, cosmo)
        assert u.isclose(value_calculated, value_expected),f"Expected {value_expected}, but got {value_calculated}"
