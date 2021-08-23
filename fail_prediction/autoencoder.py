import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class AE_architecture(nn.Module):
    """
    It implements the autoencoder neural network. It is fixed to 5 layers,
    but it can be modified to n layers.
    
    Parameters
    ----------
    channels: integer
              Number of channels to the neural network. For example, if
              only KWH will be used, then it is 1. But if KWH and current
              are used, then it is 2, etc.
    x_dim: integer
           Length of the signal.
    z_dim: integer
           Latent dimension. The autoencoder compresses the signal to
           this dimension. The smaller this number is, the higher the
           error. 

    Returns
    ------
    tensor: torch tensor
       The output of the autoencoder neural network with shape identical
       to the input shape.
    """
    def __init__(self, channels, x_dim, z_dim):
        super(AE_architecture, self).__init__()
        self.kern = 5
        self.x_dim, self.z_dim = x_dim, z_dim
        self.mu = channels
        #encoder
        self.conv1 = nn.Conv1d(self.mu, 32*self.mu, self.kern, groups=self.mu,
                               bias=False)
        self.conve = nn.ModuleList([nn.Conv1d(self.mu*2**i,self.mu*2**(i-1),self.kern,
                                              groups=self.mu, bias=False) for i in range(5, 0, -1)])
        self.fce = nn.Linear((x_dim-((self.kern-1)*6))*self.mu, self.z_dim)
        #decoder
        self.fcd = nn.Linear(self.z_dim, (x_dim-((self.kern-1)*6))*self.mu)
        self.convd = nn.ModuleList([nn.ConvTranspose1d(self.mu*2**i, self.mu*2**(i+1), self.kern,
                                                       groups=self.mu, bias=False) for i in range(5)])
        self.conv2 = nn.ConvTranspose1d(32*self.mu, self.mu, self.kern, groups=self.mu)
        self.lins = nn.ModuleList([nn.Linear(self.x_dim, self.x_dim) for i in range(self.mu)])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        for j in range(len(self.conve)):
            x = F.relu(self.conve[j](x))
        x = x.view(-1, self.mu*(self.x_dim-((self.kern-1)*6)))
        x = F.relu(self.fce(x))
        x = F.relu(self.fcd(x))
        x = x.view(-1, self.mu, self.x_dim-((self.kern-1)*6))
        for k in range(len(self.convd)):
            x = F.relu(self.convd[k](x))
        x = F.relu(self.conv2(x))
        a, b, c = x.shape
        x = torch.cat(([self.lins[i](x[:, i, :]).reshape(a, 1, c) for i in range(self.mu)]), 1)
        return x


class data_preprocessing:
    """
    This class shuffles the data and prepares it to be passed
    to the autoencoder.

    Parameters
    ----------
    func: array
          Daily consumption functioning meters.
    fail: array
          Daily consumption failing meters.
    """
    def __init__(self, func, fail):
        np.random.seed(42)
        np.random.shuffle(func)
        np.random.shuffle(fail)
        self.data = [func, fail]

    def selection(self, firsts, lasts, low, high, fail_period, percent):
        """
        This function selects the meters that were active during
        the period of interest.

        Parameters
        ----------
        firsts: list
                The first list element contains the array for
                functioning meters, second element failing meters
        lasts: list
                The first list element contains the array for
                functioning meters, second element failing meters
        low: integer
             low bound of interval
        high: integer
              high bound of interval
        fail_period: integer
                     Only meters that fail from high to high+fail_period
                     will be included in the failing meters dataset.
        percent: float (between 0 and 1)
                 What percentage of the functioning meters will be used to
                 train the neural network. If only testing, then select 0.
        """
        X = []
        for i, data in enumerate(self.data):
            cond = (lasts[i]>high) & (firsts[i]<low)
            X.append(data[cond, low:high])
            lasts[i] = lasts[i][cond]
        X[1] = X[1][lasts[1]<high+fail_period, :]
        per = int(X[0].shape[0]*percent)
        X = [x[np.sum(np.isnan(x), 1)==0, :] for x in X]
        return X[0][0:per, :], X[0][per::, :], X[1]


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
    data = data[0]
    recon = model(data)
    loss = F.mse_loss(data, recon)
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
    recon = model(data)
    loss = F.mse_loss(data, recon, reduction='none')
    loss = torch.mean(loss, dim=(1, 2))
    return loss, recon
