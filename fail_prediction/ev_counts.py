import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pdb

def first_and_last(x):
    """
    This function computes the first and last day a meter reported
    data. This function comes in handy especially for failing meters

    Parameters
    ----------
    x: array
       The raw (cumulative) KWH array. Shape is (n, m), where n
       is number of meters, and m is time.

    Returns
    -------
    firsts: ndarray
      An array of shape n that contains the first day the meters
      reported data.
    lasts: ndarray
      An array of shape n that contains the last day the meters
      reported data
    """
    length = x.shape[1]
    first = (~np.isnan(x)).argmax(axis=1)
    last = length-1-(~np.isnan(np.flip(x, 1))).argmax(axis=1)
    return first, last

    
class preprocess_data_fail_pred:
    """
    This class sets up the event data to be passed through
    a random forest classifier or logistic regression.
    """
    def __init__(self, func, fail):
        self.data = [func, fail]
        
    def select_mets_evs(self, firsts, lasts, low, high, ev_names, min_ev=0):
        """
        This method selects the meters that were active when the
        event data was collected, as well as the events that are
        nonzero for the active meters.

        Parameters
        ----------
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
        min_ev: integer (optional)
                Minimum number of event occurrences meters must have
                for the event to be included.

        Returns
        -------
        X: list
          It contains functioning meters in the first element,
          failing in the second.
        ev_names: array
          Array containing event names of events to be used.
        """
        X = []
        for i, data in enumerate(self.data):
            cond = (lasts[i]>high) & (firsts[i]<low)
            X.append(data[cond, :, :])
            lasts[i] = lasts[i][cond]
        self.lasts = lasts
        ids = np.arange(ev_names.size)
        nonzero_evs = np.sum(X[1], (0, 2))>min_ev
        ids = ids[nonzero_evs]
        for i, x in enumerate(X):
            X[i] = x[:, nonzero_evs, low:high]
        return X, ev_names[ids]

    def counting_evs(self, firsts, lasts, low, high, ev_names, fail_period):
        """
        This method calls select_mets_evs, then it further filters 
        failing to those that failed within high and high+fail_period. 
        Finally, it adds up the events in the low high time period.

        Parameters
        ----------
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
        X_u: list
          Each element contains the event count. The first list element 
          contains functioning meters, second element failing meters
        ev_names: array
          Array containing event names of events to be used.
        """
        X, ev_names = self.select_mets_evs(firsts, lasts, low, high, ev_names)
        X[1] = X[1][self.lasts[1]<high+fail_period, :, :]
        for i in range(len(X)):
            X[i] = np.sum(X[i], 2)
        return X, ev_names

    
class classifiers:
    """
    This class is used to train and evaluate the random forest and logistic
    regression algorithms. It trains on the event counts over a time period.
    """
    def __init__(self, func, fail):
        self.X_u = func
        self.X_f = fail

    def training_testing_split(self, percent):
        """
        It generates the train test split as well as the labels
        (0 for functioning and 1 for failing).

        Parameters
        ----------
        percent: float (between 0 and 1)
                 It represents the percentage that goes to testing data.
                 For example, if set to 0.8, then 80% of the data will be
                 for testing.
        
        """
        lab_u = np.zeros(self.X_u.shape[0])
        lab_f = np.ones(self.X_f.shape[0])
        X = np.concatenate((self.X_u, self.X_f), 0)
        Y = np.concatenate((lab_u, lab_f), 0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=percent, random_state=10, shuffle=True, stratify=Y)
        return X_train, X_test, y_train, y_test

    def implement_classifiers(self, X_train, X_test, y_train, y_test):
        """
        It trains random forest and logistic regression. It outputs the
        predicted probabilities.

        Parameters
        ----------
        X_train: array
                 training data.
        X_test: array
                testing data.
        y_train: array
                 training labels.
        y_test: array
                testing labels.

        Returns
        -------
        probs_LR: list
          Output probabilities of logistic regression. First element is 
          training; second element is teting.
        probs_RF: list
          Output probabilities of random forest. First element is
          training; second element is teting.
        """
        clf_LR = LogisticRegression(random_state=1, max_iter=2000)
        clf_LR.fit(X_train, y_train)
        clf_RF = RandomForestClassifier(n_estimators=40, max_depth=7, random_state=2)
        clf_RF.fit(X_train, y_train)
        prob_train_LR = clf_LR.predict_proba(X_train)
        prob_test_LR = clf_LR.predict_proba(X_test)
        prob_train_RF = clf_RF.predict_proba(X_train)
        prob_test_RF = clf_RF.predict_proba(X_test)
        probs_LR = [prob_train_LR, prob_test_LR]
        probs_RF = [prob_train_RF, prob_test_RF]
        return probs_LR, probs_RF

    
def roc_curves(probs, labels):
    """
    It calculates the roc curves for training and testing data.
    
    Parameters
    ----------
    probs: list
           Output probabilities.
    labels: list
            Labels of training and testing.

    Returns
    -------
    fprs: list
      False positive rate, aka probability of false alarm (PFA)
    tprs: list
      True positive rate, aka probability of detection (PD)
    """
    fprs = []
    tprs = []
    for i in range(len(probs)):        
        fpr, tpr, _ = metrics.roc_curve(labels[i], probs[i],
                                        pos_label =1,
                                        drop_intermediate=True)
        fprs.append(fpr)
        tprs.append(tpr)
    return fprs, tprs
        
 
