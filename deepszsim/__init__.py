"""
package for fast simulations of the Sunyaev-Zel'dovich effect from galaxy clusters
"""

import deepszsim.make_sz_cluster as make_sz_cluster
import deepszsim.dm_halo_dist as dm_halo_dist
import deepszsim.visualization as visualization
from deepszsim.load_vars import load_vars
from deepszsim.load_vars import readh5
from deepszsim.simclusters import simulate_clusters as simulate_clusters

try:
    from deepszsim.filters import get_tSZ_signal_aperture_photometry
    from deepszsim.utils import Mpc_to_arcmin
except ModuleNotFoundError:
    pass