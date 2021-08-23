import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_1D_regression(nn.Module):
    """
    It implements a CNN for 1D signals regression with a reduction (can be
    expansion) of the original signal. Architecture is: convolutional layers
    with relu, and a fully connected layer at the end and linear function.
    
    Parameters
    ----------
    x_dim: integer
        It is the length of the signal
    channels: integer
        Number of input channels
    r_factor: integer
        Reduction factor. By what value is the length of the original
        divided. If the length of the output should be half, then r_factor
        is equal to 2.
    layers: integer
        Number of convolutional layers

    Returns
    -------
    tensor: torch tensor
       The output of the neural network with shape (s, c, l), namely
       number of samples, number of channels, and length of signal

    """
    def __init__(self, x_dim, channels, r_factor,layers=5):
        super(CNN_1D_regression, self).__init__()
        self.x_dim = x_dim
        self.chan = channels
        self.r = r_factor
        self.kern = 3
        self.layers = layers
        self.conv1 = nn.Conv1d(self.chan, self.chan*4**layers, self.kern, groups=self.chan, bias=False)
        self.conv = nn.ModuleList([nn.Conv1d(self.chan*4**i, self.chan*(4**i)//4, self.kern, groups=self.chan, bias=False) for i in range(layers,0,-1)])
        self.lins = nn.ModuleList([nn.Linear(self.x_dim-(self.kern-1)*(layers+1),self.x_dim//self.r) for i in range(self.chan)])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        for j in range(self.layers):
            x = F.relu(self.conv[j](x))
        a, b, c = x.shape
        out_s = self.x_dim//self.r
        x = torch.cat(([self.lins[i](x[:, i, :]).reshape(a, 1, out_s) for i in range(self.chan)]), 1)
        return x


