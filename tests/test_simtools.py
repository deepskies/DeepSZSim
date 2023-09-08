import pytest
from simsz import simtools
import simsz.utils as utils
import scipy
import numpy as np
import astropy.constants as c
from astropy import units as u
from pixell import enmap
import camb

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

class TestSimTools():
    def test_f_sz(self):
        freq = 100
        T_CMB = cosmo.Tcmb0
        fsz = simtools.f_sz(freq,T_CMB)
        fsz_expected = 12.509
        assert u.isclose(fsz,fsz_expected), 
        f"Expected {fsz_expected}, but got {fsz}"
