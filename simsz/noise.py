import numpy as np

def generate_noise_map(N, noise_level, pix_size):
    """
    Generates a noise map based on the noise level and beam size.

    Args:
        N (int): Size of the noise map (N x N).
        noise_level (float): Noise level of the survey.

    Returns:
        ndarray: Noise map.
    """
    
    # Create random noise map
    random_noise_map = np.random.normal(0, 1, (N, N))

    # Scale random noise map by noise level
    scaled_noise_map = random_noise_map * noise_level
    noise_map = scaled_noise_map / pix_size

    return noise_map