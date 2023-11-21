import pytest
from simsz.simtools import f_sz, convolve_map_with_gaussian_beam
import simsz.utils as utils
import scipy
import numpy as np
import astropy.constants as c
from astropy import units as u
from pixell import enmap
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

class TestSimTools():
    def test_convolve_map_with_beam(self):
        pix_size_arcmin = 1.0
        beam_size_fwhp_arcmin = 3.0
        map_to_convolve = np.zeros((5, 5))
        map_to_convolve[2, 2] = 1
        convolved_map = convolve_map_with_gaussian_beam(pix_size_arcmin, beam_size_fwhp_arcmin, map_to_convolve)
        assert (np.max(convolved_map == convolved_map[0,0]))
        #max_position = np.unravel_index(np.argmax(convolved_map, axis=None), convolved_map.shape)
        #self.assertEqual(max_position, (2, 2))
        #self.assertAlmostEqual(np.sum(map_to_convolve), np.sum(convolved_map), places=5)


    def test_f_sz(self):
        freq = 1
        (cosmo,sigma8,ns) = get_mock_cosmology()
        T_CMB = cosmo.Tcmb0
        fsz = f_sz(freq,T_CMB)
        fsz_expected = -1.999948
        assert u.isclose(fsz,fsz_expected),f"Expected {fsz_expected}, but got {fsz}"
