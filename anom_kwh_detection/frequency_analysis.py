from scipy.fftpack import fft
import numpy as np


def fft_calculation(x):
    """
    This function computes the absolute value of the normalized
    frequency spectrum. The normalization is fft_i/sum(fft). 

    Parameters
    ----------
    x: array
       Shape should be (n, m), where n is the number of meters,
       and m is the length (time) of the signal.

    Returns
    -------
    array: ndarray
       The shape will be (n, m//2)
    """
    fft_x = fft(x, axis=1)
    fft_x = fft_x[:, 0:fft_x.shape[1]//2]
    fft_x_mag = np.abs(fft_x)
    fft_x_mag_freq = fft_x_mag[:, 1::]
    fft_x_norm = fft_x_mag_freq/np.sum(fft_x_mag_freq, 1)[:, np.newaxis]
    return fft_x_norm


def freq_band(fft_data, low_freq, high_freq = None):
    """
    This functions computes the power of a given frequency
    band. For anomaly detection: high_freq should be set as
    high as possible.

    Parameters
    ----------
    fft_data: array
              The normalized frequency spectrum. It has shape
              (n, m), where n is meters, and m is the length.

    low_freq: integer (smaller than m)
              The lower upper value for the frequency band.

    high_freq: integer (optional)
               The upper value for the frequency band.

    Returns
    -------
    array: ndarray
       Shape is (n)
    """
    a, b = fft_data.shape
    if high_freq is not None:
        high = high_freq
    else:
        high = b
    return np.sum(fft_data[:, low_freq:high], 1)
    


