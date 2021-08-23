import numpy as np
import scipy.ndimage as ndimage


def subsampling(x, factor, mean=True):
    """
    This function subsamples the signal with the mean or median.

    Parameters
    ----------
    x : array
        Array should have size (n,m), where n is the number of
        samples (meters), and m is the length.
    factor: integer
            The factor by which the signal will be reduced. It
            must divide the length of the signal.
    mean: bool (optional)
          If True, the mean will be used; else median.

    Returns
    -------
    array: ndarray
       The array will have shape (n,m//factor+2) because the
       edge values are mirrored.
    """
    x = x.reshape((x.shape[0], -1, factor))
    if mean == True:
        x = np.mean(x, axis=2)
    else:
        x = np.median(x, axis=2)
    x = np.pad(x, ((0, 0), (1, 1)), mode='edge')
    return x


def SD_coefficient(x):
    """
    This function computes the SD coefficient, as defined in
    equation 1 in the report.

    Parameters
    ----------
    x : array
        Array should have size (n,m), where n in the number of
        samples (meters), and m is the length

    Returns
    -------
    array: ndarray
       It returns the n SD scores.
    """
    length = x.shape[1]
    derivative = ndimage.convolve1d(x, weights=[-1,1], axis=1)
    SD = (x[:,0]-x[:,-1])/np.sum(np.abs(derivative),1)
    SD = length*SD
    return SD
    
