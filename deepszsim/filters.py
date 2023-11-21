"""
submap filtering tools to analyze SZ submaps
"""

import numpy as np

def get_tSZ_signal_aperture_photometry(dT_map, radmax_arcmin, 
                                           fmax_arcmin=np.sqrt(2)):
    """
    Parameters:
    ----------
    dT_map: array to represent the map in uK
    radmax_arcmin: float
        the radius of the inner aperture, in arcmin
    fmax_arcmin: float
        fmax+radmax is the radius of the outer aperture, in arcmin

    Returns:
    -------
    disk_mean: float
        The average value within an annulus of inner radius R
    ring_mean: float
        The average value within an annulus of outer radius sqrt(2)*R
    tSZ signal: float
        thermal SZ effect signal
    """

    center = np.array(dT_map.shape) / 2
    x, y = np.indices(dT_map.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    #define outer radius
    radius_out=radmax_arcmin * fmax_arcmin 
    #average in inner radius
    disk_mean = dT_map[r < radmax_arcmin].mean() 
    #average in outer radius
    ring_mean = dT_map[(r >= radmax_arcmin) & (r < radius_out)].mean()
    tSZ = disk_mean - ring_mean

    return disk_mean, ring_mean, tSZ
