"""
functions to generate simulated CMB map noise
"""

import numpy as np

def generate_noise_map(image_size, noise_level, pix_size):
    """
    Generates a white noise map based on the noise level and beam size.

    Args:
        image_size: int
            Size of the noise map (N x N).
        noise_level: float
            Noise level of the survey.
        pix_size: int
            size of pixels in arcminutes

    Returns:
        ndarray: Noise map.
    """
    
    # Create random noise map
    random_noise_map = np.random.normal(0, 1, (image_size, image_size))

    # Scale random noise map by noise level
    scaled_noise_map = random_noise_map * noise_level
    noise_map = scaled_noise_map / pix_size
    

    return noise_map
