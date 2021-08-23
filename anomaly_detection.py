import numpy as np
#from . import nn_train_test
#from .anom_kwh_detection import general_anomalies as ga
#from .anom_kwh_detection import slowing_down_meters as sdm
#from .anom_kwh_detection import frequency_analysis as fa
import nn_train_test
from anom_kwh_detection import general_anomalies as ga
from anom_kwh_detection import slowing_down_meters as sdm
from anom_kwh_detection import frequency_analysis as fa


def nn_anomaly_detection(data, model):
    """
    This function computes the anomaly score using neural networks
    as described in the report.

    Parameters
    ----------
    data: array
          This should be the clean KWH signal.
    model: pytorch model
           This is the trained neural network.

    Returns
    -------
    output: array
            This is the prediction of the neural network
    score: array
           This array contains the anomaly scores for all
           meters throughout time.
    """
    data = [data[:, np.newaxis, :]]
    nn_model = nn_train_test.nn_model(model)
    loss, output = nn_model.test(data, ga.testing)
    test_data = data[0][:, 0, 28::]
    delta = ga.nn_anomaly_score(output, test_data)
    return output, delta


def slowing_down_meters_detection(data, factor):
    """
    This function computes the SD score as described in the
    report

    Parameters
    ----------
    data: array
          This should be the clean KWH signal.
    factor: integer
            The factor by which data will be subsampled.
            The length of the data must be divisible by factor.

    Returns
    -------
    array: ndarray
       The SD scores computed for every meter.
    """
    subsampled = sdm.subsampling(data, factor)
    SD = sdm.SD_coefficient(subsampled)
    return SD


def frequency_based_detection(data, low_freq, high_freq=None):
    """
    This function computes the high frequency power.

    Parameters
    ----------
    data: array
          Shape should be (n, m), where n is the number of meters,
          and m is the length (time) of the signal.
    low_freq: integer (smaller than m//2)
              The lower upper value for the frequency band.

    high_freq: integer (optional)
               The upper value for the frequency band.

    Returns
    -------
    array: ndarray
       shape is (n)
    """
    fft_data = fa.fft_calculation(data)
    power = fa.freq_band(fft_data, low_freq, high_freq=high_freq)
    return power