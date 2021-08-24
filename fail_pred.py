import numpy as np
#from . import nn_train_test
#from .fail_prediction import ev_counts
#from .fail_prediction import autoencoder as AE
import nn_train_test
from fail_prediction import ev_counts
from fail_prediction import autoencoder as AE
from itertools import chain



def event_counting(data, firsts, lasts, low, high, ev_names, fail_period):
    """
    This function trains and evaluates a failure prediction algorithms
    based on random forest (RF) and logistic regression(LR). The output is
    the roc curves for training and testing of the respective approaches.

    Parameters
    ----------
    data: list
          First element should have functioning meters. Second
          element should have failing meters.
    firsts: list
            The first list element contains the array for
            functioning meters, second element failing meters
    lasts: list
            The first list element contains the array for
            functioning meters, second element failing meters
    low: integer
         low bound of interval to count events on
    high: integer
          high bound of interval to count events on
    ev_names: array
              Array containing event names of events to be used.
    fail_period: integer
                 Only meters that fail from high to high+fail_period
                 will be included in the failing meters dataset.
    
    Returns
    -------
    fprs: list
          False positive rates (PFA). The order is: 0: Train LR; 
          Test LR; Train RF; Test RF.
    tprs: list
          True positive rates (PD). The same order as fprs.
    """
    Data = ev_counts.preprocess_data_fail_pred(data[0], data[1])
    X, ev_names = Data.counting_evs(firsts, lasts, low, high,
                                    ev_names, fail_period)
    models = ev_counts.classifiers(X[0], X[1])
    X_train, X_test, y_train, y_test = models.training_testing_split(.4)
    probs_LR, probs_RF = models.implement_classifiers(X_train, X_test,
                                                      y_train, y_test)
    probs_all = list(chain(*[probs_LR, probs_RF]))
    probs_all = [probs_all[i][:, 1] for i in range(len(probs_all))]
    labels = [y_train, y_test]
    labels_all = list(chain(*[labels, labels]))
    fprs, tprs = ev_counts.roc_curves(probs_all, labels_all)
    return fprs, tprs


def AE_fail_prediction(data, model, firsts, lasts, low, high,
                       fail_period, percent, train=False):
    """
    This function uses the autoencoder approach to get
    roc curves for the failure prediction problem.
    
    Parameters
    ----------
    data: list
          First element should have functioning meters. Second
          element should have failing meters.
    firsts: list
            The first list element contains the array for
            functioning meters, second element failing meters
    lasts: list
            The first list element contains the array for
            functioning meters, second element failing meters
    low: integer
         low bound of interval to count events on
    high: integer
          high bound of interval to count events on
    ev_names: array
              Array containing event names of events to be used.
    fail_period: integer
                 Only meters that fail from high to high+fail_period
                 will be included in the failing meters dataset.
    percent: float (between 0 and 1)
             The percentage of the functioning meters will be used to
             train the neural network. If only testing, then select 0.
    train: bool (optional)
           Select True if model requires training.

    Returns
    -------
    fprs: list
          False positive rates (PFA). The order is: 0: Train LR; 
          Test LR; Train RF; Test RF.
    tprs: list
          True positive rates (PD). The same order as fprs.
    """
    Data = AE.data_preprocessing(data[0], data[1])
    U_train, U_test, F_test = Data.selection(firsts, lasts, low,
                                             high, fail_period, percent)
    U_train, U_test = [U_train[:, np.newaxis, :]], [U_test[:, np.newaxis, :]]
    F_test = [F_test[:, np.newaxis, :]]
    nn_model = nn_train_test.nn_model(model)
    if train==True:
        path_model = 'nn_models/autoencoder.pth'
        train_loss = nn_model.train(U_train, 1000, 0.001, AE.trainig,
                                    save=True, path=path_model)
    loss_u, output_u = nn_model.test(U_test, AE.testing)
    loss_f, output_f = nn_model.test(F_test, AE.testing)
    losses = [np.concatenate((loss_u, loss_f), 0)]
    labels = [np.concatenate((np.zeros(loss_u.size), np.ones(loss_f.size)), 0)]
    fprs, tprs = ev_counts.roc_curves(losses, labels)
    return fprs, tprs
