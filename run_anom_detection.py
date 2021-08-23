import numpy as np
from anomalous_kwh_detection import nn_architectures
import argparse
import utils
import torch
import anomaly_detection as an

DAYS = 672

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Filename that contains KWH', nargs="?")
args = parser.parse_args()
path = 'data/'


data = np.load(path+args.filename)
data = data[:, 0:DAYS]


data = utils.data_prepocessing(data)

model_3_layers = nn_architectures.CNN_1D_regression(28, 1, 4, layers=3)
model_5_layers = nn_architectures.CNN_1D_regression(28, 1, 4, layers=5)

model_3_state_dict = torch.load('nn_models/3_layers_state_dict.pth')
model_5_state_dict = torch.load('nn_models/5_layers_state_dict.pth')

model_3_layers.load_state_dict(model_3_state_dict)
model_5_layers.load_state_dict(model_5_state_dict)

out_3, delta_3 = an.nn_anomaly_detection(data, model_3_layers)
out_5, delta_5 = an.nn_anomaly_detection(data, model_5_layers)
test_data = data[:, 28::]

SD_7 = an.slowing_down_meters_detection(data, 7)
SD_14 = an.slowing_down_meters_detection(data, 14)

freq_250 = an.frequency_based_detection(data, 250)
freq_300 = an.frequency_based_detection(data, 300)

data_nns = np.concatenate((out_3[:, :, np.newaxis], out_5[:, :, np.newaxis], test_data[:, :, np.newaxis]), axis=2)
errs_nns = np.concatenate((delta_3[:, :, np.newaxis], delta_5[:, :, np.newaxis]), axis=2)

SDs = np.concatenate((SD_7[:, np.newaxis], SD_14[:, np.newaxis]), axis=1)

freqs = np.concatenate((freq_250[:, np.newaxis], freq_300[:, np.newaxis]), axis=1)

path_results = 'results/'
np.save(path_results+'data_nns.npy', data_nns)
np.save(path_results+'errs_nns.npy', errs_nns)
np.save(path_results+'SDs.npy', SDs)
np.save(path_results+'freqs.npy', freqs)
np.save(path_results+'data.npy', data)
