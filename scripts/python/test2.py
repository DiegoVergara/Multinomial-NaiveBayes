from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import issparse

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import binarize
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot, logsumexp
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.fixes import in1d
from numpy.core.umath_tests import inner1d
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
from sklearn.metrics import roc_curve,auc,confusion_matrix,classification_report

class MultinomialNB():

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        #self.fit_prior = fit_prior
        #self.class_prior = class_prior

    # def _get_coef(self):
    #     return (self.feature_log_prob_[1:]
    #             if len(self.classes_) == 2 else self.feature_log_prob_)

    # def _get_intercept(self):
    #     return (self.class_log_prior_[1:]
    #             if len(self.classes_) == 2 else self.class_log_prior_)

    def fit(self, X, y, sample_weight=None):

        #X, y = check_X_y(X, y, 'csr')
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        Y = Y.astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= check_array(sample_weight).T

        #class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features), dtype=np.float64)
        #if np.any((X.data if issparse(X) else X) < 0):
        #    raise ValueError("Input X must be non-negative")
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1)
        
        self.feature_log_prob_ = (np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1)))
        #self._update_feature_log_prob()
        #self._update_class_log_prior(class_prior=class_prior)
        n_classes = len(self.classes_)
        self.class_log_prior_ = np.zeros(n_classes) - np.log(n_classes)
        return self


    # def _update_class_log_prior(self, class_prior=None):
    #     n_classes = len(self.classes_)
    #     if class_prior is not None:
    #         if len(class_prior) != n_classes:
    #             raise ValueError("Number of priors must match number of"
    #                              " classes.")
    #         self.class_log_prior_ = np.log(class_prior)
    #     elif self.fit_prior:
    #         # empirical prior, with sample_weight taken into account
    #         self.class_log_prior_ = (np.log(self.class_count_) - np.log(self.class_count_.sum()))
    #     else: # here
    #         self.class_log_prior_ = np.zeros(n_classes) - np.log(n_classes)

    # def _count(self, X, Y):
    #     """Count and smooth feature occurrences."""
    #     if np.any((X.data if issparse(X) else X) < 0):
    #         raise ValueError("Input X must be non-negative")
    #     self.feature_count_ += safe_sparse_dot(Y.T, X)
    #     self.class_count_ += Y.sum(axis=0)

    # def _update_feature_log_prob(self):
    #     """Apply smoothing to raw counts and recompute log probabilities"""
    #     smoothed_fc = self.feature_count_ + self.alpha
    #     smoothed_cc = smoothed_fc.sum(axis=1)

    #     self.feature_log_prob_ = (np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1)))


	# def partial_fit(self, X, y, classes=None, sample_weight=None):
	# 	X = check_array(X, accept_sparse='csr', dtype=np.float64)
	# 	_, n_features = X.shape


 #        self.coef_ = self._get_coef()
 #        #self.intercept_ = self._get_intercept()

 #        if _check_partial_fit_first_call(self, classes):
 #            # This is the first call to partial_fit:
 #            # initialize various cumulative counters
 #            n_effective_classes = len(classes) if len(classes) > 1 else 2
 #            self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
 #            self.feature_count_ = np.zeros((n_effective_classes, n_features), dtype=np.float64)
 #        elif n_features != self.coef_.shape[1]:
 #            msg = "Number of features %d does not match previous data %d."
 #            raise ValueError(msg % (n_features, self.coef_.shape[-1]))

 #        Y = label_binarize(y, classes=self.classes_)
 #        if Y.shape[1] == 1:
 #            Y = np.concatenate((1 - Y, Y), axis=1)

 #        n_samples, n_classes = Y.shape

 #        if X.shape[0] != Y.shape[0]:
 #            msg = "X.shape[0]=%d and y.shape[0]=%d are incompatible."
 #            raise ValueError(msg % (X.shape[0], y.shape[0]))

 #        # label_binarize() returns arrays with dtype=np.int64.
 #        # We convert it to np.float64 to support sample_weight consistently
 #        Y = Y.astype(np.float64)
 #        if sample_weight is not None:
 #            sample_weight = np.atleast_2d(sample_weight)
 #            Y *= check_array(sample_weight).T

 #        class_prior = self.class_prior

 #        # Count raw events from data before updating the class log prior
 #        # and feature log probas
 #        self._count(X, Y)

 #        # XXX: OPTIM: we could introduce a public finalization method to
 #        # be called by the user explicitly just once after several consecutive
 #        # calls to partial_fit and prior any call to predict[_[log_]proba]
 #        # to avoid computing the smooth log probas at each call to partial fit
 #        self._update_feature_log_prob()
 #        self._update_class_log_prior(class_prior=class_prior)
 #        return self

    def predict(self, X):

        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse='csr')
        return (safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_)


    def predict_log_proba(self, X):

        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):

        #return np.exp(self.predict_log_proba(X))
        
        jll = self._joint_log_likelihood(X)
        max_value = np.amax(jll)
        min_value = np.amin(jll)
        jll = (jll-min_value)/(max_value-min_value)
        #return np.exp(jll)
        return jll
        


class adaboost_mnb:

    def __init__(self, _data, _labels, _data_test, _n_stimators, _learning_rate=1.0, _alpha=1.0):
        self.MNB_alpha=_alpha
        self.labels=_labels
        self.data=_data
        self.data_test=_data_test
        self.n_estimators=_n_stimators
        self.learning_rate=_learning_rate
        self.dim=self.data.shape[1]
        self.n_data=self.data.shape[0]
        self.classes=np.unique(self.labels)
        self.n_classes=len(self.classes)


    def samme_proba(self, estimator):

        proba = estimator.predict_proba(self.data_test)

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = proba
        #log_proba = np.log(proba)

        return (self.n_classes - 1) * (log_proba - (1. / self.n_classes) * log_proba.sum(axis=1)[:, np.newaxis])
       
        
    def boost(self,iteration, w):
        
        #clf = GaussianNB()
        clf = MultinomialNB(alpha=self.MNB_alpha, class_prior=None, fit_prior=False)

        clf.fit(self.data, self.labels, sample_weight=w)
        #######
        Y_predict_proba = clf.predict_proba(self.data)
        #print Y_predict_proba 
        
        Y_hat = self.classes.take(np.argmax(Y_predict_proba, axis=1), axis = 0)

        index = np.where(self.labels!=Y_hat, 1, 0)
        #index = self.labels!=Y_hat

        e = np.sum(w*index)/np.sum(w)
        if e <= 0:
            print "e negativo"
            return w, 1., 0., clf
        
        # if e >= 1. - (1. / self.n_classes):
        #     print "error "
        #     return None, None, None, clf

        y_codes = np.array([-1. / (self.n_classes -1.), 1.])
        y_coding = y_codes.take(self.classes == self.labels[:, np.newaxis])

        #proba = Y_predict_proba
        #proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        #print Y_predict_proba;

        #alpha = (-1. * self.learning_rate * (((self.n_classes -1.) / self.n_classes) * inner1d(y_coding, np.log(Y_predict_proba))))
        alpha = (-1. * self.learning_rate * (((self.n_classes -1.) / self.n_classes) * inner1d(y_coding, Y_predict_proba)))
        if not iteration == self.n_estimators - 1:
            w = w* np.exp(alpha *((w > 0) | (alpha < 0)))
            #w = w* np.exp(alpha)
        #######
        return w, 1., e, clf


    def fit(self):

        w = np.ones((self.n_data),dtype=float)/self.n_data
        self.alphas = np.zeros(self.n_estimators, dtype=np.float)
        errors = np.ones(self.n_estimators, dtype=np.float)

        self.classifiers = []

        for i in xrange(0,self.n_estimators):
            #print i
            w, self.alphas[i], errors[i], clf = self.boost(i, w) ###### Boost

            w_sum = np.sum(w)  

            if (w is None) or (errors[i] == 0) or (w_sum <= 0):
                print "break"
                break
                
            self.classifiers.append(clf)
            
            if i < n_estimators - 1:
                    w /= w_sum

    def predict(self):

        self.classes = self.classes[:, np.newaxis]
        pred = None
        
        # pred = sum((estimator.predict(self.data_test) == self.classes).T * weights 
        #                     for estimator, weights in zip(self.classifiers,self.alphas))

        pred = sum(self.samme_proba(estimator) for estimator in self.classifiers)

        pred /= self.alphas.sum()
        if self.n_classes == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)

        if self.n_classes == 2:
                return self.classes.take(pred > 0, axis=0)

        return self.classes.take(np.argmax(pred, axis=1), axis=0)


"""
n = 14000

data=np.loadtxt('../data/adience/dataset.csv',delimiter=',')
classes=np.loadtxt('../data/adience/gender_label.csv',delimiter=',')

data_train = data[0:n,:]
data_test = data[n:,:]
classes_train = classes[0:n]
classes_test = classes[n:]
"""
"""
data_train = np.array([[6,180,12],[5.92,190,11],[5.58,170,12],[5.92,165,10],[5,100,6],[5.5,150,8],[5.42,130,7],[5.75,150,9]])

data_test = np.array([[6,130,8]])

classes_train = np.array([1,
        1,
        1,
        1,
        0,
        0,
        0,
        0])

classes_test =np.array([0])
"""
"""
n_data=data_train.shape[0]
w = np.ones((n_data),dtype=float)/n_data

clf = MultinomialNB(alpha=1, class_prior=None, fit_prior=False)
clf.fit(data_train, classes_train, sample_weight=None)

#Y_predict_proba = clf.predict_proba(data_test)
Y_hat = clf.predict(data_test)
#print Y_predict_proba 

print(classification_report(classes_test , Y_hat))
"""

n = 14000

data=np.loadtxt('../../data/adience/dataset.csv',delimiter=',')
classes=np.loadtxt('../../data/adience/gender_label.csv',delimiter=',')

data_train = data[0:n,:]
data_test = data[n:,:]
classes_train = classes[0:n]
classes_test = classes[n:]

n_estimators = 100
learning_rate = 1.0
alpha = 1.0


ada_mnb=adaboost_mnb(data_train, classes_train, data_test, n_estimators, learning_rate, alpha)
ada_mnb.fit()
Y_hat = ada_mnb.predict()
print(classification_report(classes_test , Y_hat))
