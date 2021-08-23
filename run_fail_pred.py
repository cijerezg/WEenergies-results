import numpy as np
from fail_prediction import autoencoder as AE
from fail_prediction import ev_counts
import fail_pred as fp
import utils
import argparse
import sparse
import pandas as pd

path = 'data/'

met_type = 'bus_N'

meters_dict = {'res_N':2,'bus_N':3, 'bus_B':5}

evs_f = sparse.load_npz(path+'fail_events.npz')
reads_f = np.load(path+'fail_reads.npy')
fail_ids = np.load(path+'fail_types.npy')

evs_f = evs_f[fail_ids==meters_dict[met_type], :, :]
evs_f = evs_f.todense()
reads_f = reads_f[fail_ids==meters_dict[met_type], :]

ev_names = pd.read_csv(path+'ev_names.csv', header=None)
ev_names = np.asarray(ev_names)

evs_u = sparse.load_npz(path+met_type+'_events.npz')
evs_u = evs_u.todense()
reads_u = np.load(path+met_type+'_reads.npy')


#Event counting approach
data = [evs_u, evs_f]

first_u, last_u = ev_counts.first_and_last(reads_u)
first_f, last_f = ev_counts.first_and_last(reads_f)
firsts = [first_u, first_f]
lasts = [last_u, last_f]

LOW = 300
HIGH = 390
FAIL_PERIOD = 120


fprs_evs, tprs_evs = fp.event_counting(data, firsts, lasts, LOW,
                                       HIGH, ev_names, FAIL_PERIOD)

labels = ['Train LR', 'Test LR', 'Train RF', 'Test RF']
cols = ['tab:blue','tab:blue', 'tab:orange', 'tab:orange']
linestyle = ['--', 'solid', '--', 'solid']
utils.plot_roc_curves(tprs_evs, fprs_evs, labels, cols, linestyle, 'ev_counts')


#Autoencoder approach
reads_u = utils.data_prepocessing(reads_u, indexes=False)
reads_f = utils.data_prepocessing(reads_f, indexes=False)
reads = [reads_u, reads_f]

LOW_AE = 366
HIGH_AE = 398
FAIL_PERIOD = 120
PERCENT = .5

model = AE.AE_architecture(1, 32, 4)

firsts = [first_u, first_f]
lasts = [last_u, last_f]

fprs_kwh, tprs_kwh = fp.AE_fail_prediction(reads, model, firsts, lasts, LOW_AE,
                                           HIGH_AE, FAIL_PERIOD,.5, train=True)
labels = ['Autoencoder']
cols = ['tab:orange']
linestyle = ['solid']
utils.plot_roc_curves(tprs_kwh, fprs_kwh, labels, cols, linestyle, 'AE')
