import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd


def trainig(data, model):
    """
    This function is used wihtin a class, not by itself. 
    It defines the training function required to train 
    a neural network.
    
    Parameters
    ----------
    data: array
          Array should have size (n,c,m), where n is the number
          of samples (meters), c is the number of channels (1 in
          this case), and m is the length
    model: pytorch model
          Extension to the file is .pt
    """
    loss = 0
    data = data[0]
    for i in range(data.shape[2]//7-4):
        out = model(data[:, :, 7*i:7*i+28])
        loss = loss + F.mse_loss(data[:, :, 7*i+28:7*i+35], out)
    return loss
    

def testing(data, model):
    """
    This function is used wihtin a class, not by itself.
    It is used on new data to be tested

    Parameters
    ----------
    data: array
          Array should have size (n,c,m), where n is the number
          of samples (meters), c is the number of channels (1 in
          this case), and m is the length
    model: pytorch model
          Extension to the file is .pt
    """
    data = data[0]
    outs = []
    losses = []
    for i in range(data.shape[2]//7-4):
        out = model(data[:, :, 7*i:7*i+28])
        loss = F.mse_loss(data[:, :, 7*i+28:7*i+35], out, reduction='none')
        loss = torch.mean(loss, (1, 2))
        out = out.squeeze()
        outs.append(out)
        losses.append(loss)
    outs = torch.cat(([outs[i] for i in range(len(outs))]), 1)
    losses = torch.cat(([losses[i].reshape(-1, 1) for i in range(len(losses))]), 1)
    return losses, outs


def kernel(base=0.95, length=28):
    """
    This function computes the kernel used to calculate
    the delta error.

    Parameters
    ----------
    base : float (optional)
           This is the base that will be exponentiated. It
           should be between 0 and 1.
    length : integer (optional)
           This is the length of the kernel.

    Returns
    -------
    array: ndarray
       The array will have shape (length).
    """
    kern = []
    for i in range(length):
        kern.append(np.power(base, i))
    kern = np.asarray(kern)
    return kern/np.sum(kern)


def delta_err(error, kernel):
    """
    This function calculates the delta error and normalizes
    with the median across time.

    Parameters
    ----------
    error: array
           Squared Error between the predicted signal
           the original signal.
    kernel: array
            Kernel computed using the function kernel

    Returns
    -------
    array: ndarray
       The delta error array with shape (n,m-l)
    """
    delta = []
    for i in range(kernel.size+1, error.shape[1]):
        err = np.abs(error[:, i]-np.dot(error[:, i-kernel.size:i], kernel))
        delta.append(err)
    delta = np.asarray(delta)
    delta = delta/np.median(delta, 0)[np.newaxis, :]
    return np.transpose(delta)


def nn_anomaly_score(out, data, length = 28, base = 0.95):
    """
    This function computes the anomaly score for meters

    Parameters
    ----------
    out: array
         This is the neural network output.
    data:xo array
          This is the original data. It is crucial that
          it corresponds to the same time period as the
          neural network.
    length: integer (optional)
            This will establish the length of the kernel.
    base: float (optional, between 0 and 1)
          This will establish the base of the kernel.

    Returns
    -------
    array: ndarray
       It returns an array with shape (n,m), where n is 
       the number of meters, m is refers to the time axis.
    """
    square_err = np.power(out-data, 2)
    kern = kernel(base=base, length=length)
    delta = delta_err(square_err, kern)
    delta = delta/np.median(delta, 0)[np.newaxis, :]
    delta = pd.DataFrame(delta)
    delta = delta.rolling(7, min_periods=1, axis=1).mean()
    delta = np.asarray(delta)
    return delta

