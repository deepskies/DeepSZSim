"""
plotting functions
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm
from deepszsim.utils import arcmin_to_Mpc

def plot_graphs(image, title = None, xlabel = None, ylabel = None, cbarlabel = None, width = None, specs = None,
                extend = False,
                logNorm = False):
    '''
    Plotting tool function for our 2D submaps and CMB maps. 
    
    Parameters:
    -----------
    image - float array
        the graph we are plotting
    title - str
        title of the graph
    xlabel - str
        label of the x-axis
    ylabel - str
        label of the y-axis
    cbarlabel - str
        label of the color bar
    width: int
        half rounded down of the width of output plot in pixels (eg, image size = 2*width+1)
    specs: Optional[dict]
        optional dictionary to pass title, xlabel, ylabel, cbarlabel, and width
    logNorm: bool
        if true, uses a logarithmic normalization for the plot (using SymLogNorm in case values are negative)
    

    Returns:
    -------
    none
    '''
    
    if specs is not None:
        title, xlabel, ylabel, cbarlabel, width = specs['title'], specs['xlabel'], specs['ylabel'], \
                                                  specs['cbarlabel'], specs['width']
    if logNorm:
        if np.min(image)<0:
            imgflatabs = np.abs(image.flatten())
            im = plt.imshow(image, norm = SymLogNorm(linthresh =  np.min(imgflatabs[np.nonzero(imgflatabs)])))
        else:
            im = plt.imshow(image, norm=LogNorm())
    else:
        im = plt.imshow(image)
    cbar = plt.colorbar(im)
    im.set_extent([-width,width,-width,width])
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    cbar.set_label(cbarlabel, rotation=270)


def plotting_specs(cluster):
    """
    
    Args:
        cluster:
            dictionary representing a cluster instance, which contains a `parameters` dictionary which itself has 'M200', 'redshift', and 'image_size_pixels' keys

    Returns:
        dictionary that can be passed as the value of the `specs` kwarg in `plot_graphs`
    """
    out = {}
    out['title'] = f"M200 = {cluster['params']['M200']:.2e}, z = {cluster['params']['redshift_z']:.3f}"
    out['xlabel'] = "arcmin"
    out['ylabel'] = "arcmin"
    out['cbarlabel'] = "Compton y"
    out['width'] = (cluster['params']['image_size_pixels']-1)//2
    return out
