"""
package for fast simulations of the Sunyaev-Zel'dovich effect from galaxy clusters
"""

import simsz.make_sz_cluster as make_sz_cluster
import simsz.dm_halo_dist as dm_halo_dist
import simsz.visualization as visualization
from simsz.load_vars import load_vars
from simsz.load_vars import readh5

try:
    from simsz.filters import get_tSZ_signal_aperture_photometry
    from simsz.utils import Mpc_to_arcmin
except ModuleNotFoundError:
    pass