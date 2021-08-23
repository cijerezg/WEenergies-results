import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 28})


def daily_consumption(x):
    """
    This function takes the cumulative consumption
    and computes the daily consumption.

    Parameters
    ----------
    x: array
       The shape should be (n,m), where n is the number
       of meters, and m the length.

    Returns
    -------
    array: ndarray
       The daily consumption array whose shape is the 
       same as the original array.
    """
    aux = np.zeros(x.shape)
    for k in range(1, x.shape[1]):
        past = x[:, k-1]
        fut = x[:, k]
        aux[:, k] = np.where((past>80000)&(fut<15000),100000-past+fut,fut-past)
    aux[:, 0] = aux[:, 1]
    return aux


class ids_select:
    """
    This function does data cleaning by selecting the
    meter indexes that satisfy the data cleaning conditions.
    """
    def __init__(self, data):
        self.data = data
        self.index = np.arange(data.shape[0])

    def nans(self):
        """
        This method enforces the no nan condition.
        """
        return self.index[np.sum(np.isnan(self.data), 1)==0]        

    def neg_kwh(self):
        """
        This method enforces that there is no negative.
        KWH consumption
        """
        return self.index[np.nanmin(self.data, 1) > 0]

    def large(self):
        """
        This method enforces that unrealistically large.
        KWH consumptions are excluded
        """
        return self.index[np.nanmax(self.data, 1) < 2000]

    def indexes(self, nan=True, neg=True, lar=True):
        """
        This method collects all the indexes from the conditions
        and calculates their intersection.
        """
        inds = []
        if nan==True:
            inds.append(self.nans())
        if neg==True:
            inds.append(self.neg_kwh())
        if lar==True:
            inds.append(self.large())
        return list(set.intersection(*map(set, inds)))


def data_prepocessing(data, shuffle=True, indexes=True):
    """
    This function takes the raw cumulative KWH and outputs
    the clean daily KWH.

    Parameters
    ----------
    data: array
          This array has shape (n,m), where n is the number of
          meters, and m is time.
    shuffle: bool (optional)
             Indicates whether the meters will be shuffled or not.
    indexes: bool (optional)
             Indicates whether certain meters are excluded or not.

    Returns
    -------
    array: ndarray
       Generally the shape will be (p,m), where p<n because some
       meters were discarded.

    """
    np.random.seed(42)
    data = daily_consumption(data)
    data = pd.DataFrame(data)
    data = data.rolling(3, min_periods=1, axis=1).mean()
    data = np.asarray(data)
    if indexes==True:
        ids = ids_select(data)
        index = ids.indexes()
        data = data[index, :]
    np.random.shuffle(data)
    return data


def normalization_0_1(x):
    """
    It normalizes between 0 and 1 along the last axis of array

    Parameters
    ----------
    x : array
        Array can have any size, and normalization occurs in the last
        axis.
    
    Returns
    -------
    array: ndarray
       The normalized array between 0 and 1.
    """
    dim = len(x.shape)
    minv = np.min(x, axis=dim-1)
    num = x-np.expand_dims(minv, axis=dim-1)
    den = np.expand_dims(np.max(x, axis=dim-1)-minv, axis=dim-1)
    return num/den


def plot_roc_curves(tprs, fprs, labels, cols, linestyle, name):
    """
    This function plots roc curves and saves the image file to
    the folder figures.

    Parameters
    ----------
    tprs: list
          True positive rates (PD)
    fprs: list
          False positive rates (PFA)
    labes: list
           Labels assigned to each curve in tprs and fprs
    cols: list
          Colors assigned to each curve in tprs and fprs
    linestyle: list
               Linestyles assigned to each curve in tprs and fprs
    name: string
          name of the image. It is saved in the folder figures
    """
    fig = plt.figure(figsize=(15,13))
    for i in range(len(tprs)):
        plt.plot(fprs[i], tprs[i], label=labels[i], color = cols[i],
                 linestyle=linestyle[i], linewidth=2.5)
    plt.plot([0, 1] ,[0, 1], label='Random', c='purple')
    plt.grid(True)
    plt.xlabel('PFA')
    plt.ylabel('PD')
    plt.legend()
    plt.savefig('figures/'+name, bbox_inches='tight')
    plt.close()
