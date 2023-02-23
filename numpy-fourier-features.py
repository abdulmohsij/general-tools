import numpy as np

def input_array_to_fourier_features(x, B):
    '''
    Adapted for numpy from:
    https://github.com/tancik/fourier-feature-networks
    
    Params:
    x - A numpy array containing input features into a model e.g 3D coordinates would be of size (N,M) where M=3
    B - A random 'tall skinny' matrix such as from a gaussian distribution e.g np.random.normal(size=(K,M)) where K can be 256

    M features are mapped to K features. In the example above 3 features (x,y,z) will be mapped to 256 high-dimensional features.

    Learning resolution can be controlled by scaling B such as np.random.normal(size=(K,M)) * 10
    Read https://arxiv.org/abs/2006.10739 for more details.

    Returns: np.array mapped to fourier features
    '''
    if B is None:
        return x
    else:
        x_proj = (2.*np.pi*x) @ B.T
        return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)