"""
submap filtering tools to analyze SZ submaps
"""

import numpy as np

def get_tSZ_signal_aperture_photometry(dT_map, radmax_arcmin, pixel_scale, 
                                       fmax=np.sqrt(2)):
    """
    Retrieve aperture photometry of the tSZ signal
    
    Parameters:
    ----------
    dT_map: float array
        map in uK
    radmax_arcmin: float
        radius of the inner aperture in arcmin
    pixel_scale: float
        arcmin per pixel for the current settings
    fmax: float
        fmax * radmax_arcmin is the radius of the outer aperture in arcmin

    Returns:
    -------
    disk_mean: float
        average value within an annulus of inner radius r
    ring_mean: float
        average value within an annulus of outer radius sqrt(2)*r
    tSZ signal: float
        thermal SZ effect signal
    """

    radmax_pixels = radmax_arcmin / pixel_scale
    radius_out_pixels = radmax_pixels * fmax
    
    center = np.array(dT_map.shape) // 2
    x, y = np.indices(dT_map.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
    disk_mean = dT_map[r < radmax_pixels].mean()
    ring_mean = dT_map[(r >= radmax_pixels) & (r < radius_out_pixels)].mean()
    tSZ = disk_mean - ring_mean
    return disk_mean, ring_mean, tSZ
