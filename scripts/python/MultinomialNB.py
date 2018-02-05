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
        self.fit_prior = fit_prior
        self.class_prior = class_prior


    def fit(self, X, y, sample_weight=None):

        X, y = check_X_y(X, y, 'csr')
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

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features), dtype=np.float64)

        #self._count(X, Y)
        #self._update_feature_log_prob()
        #self._update_class_log_prior(class_prior=class_prior)
        
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        print "feature_count_"
        print self.feature_count_

        self.class_count_ += Y.sum(axis=0)
        print "class_count_"
        print self.class_count_

        smoothed_fc = self.feature_count_ + self.alpha
        print "smoothed_fc"
        print smoothed_fc

        smoothed_cc = smoothed_fc.sum(axis=1)
        print "smoothed_cc"
        print smoothed_cc

        self.feature_log_prob_ = (np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1)))
        print "feature_log_prob_"
        print self.feature_log_prob_

        n_classes = len(self.classes_)
        self.class_log_prior_ = np.zeros(n_classes) - np.log(n_classes)
        print "class_log_prior_"
        print self.class_log_prior_
        
        return self

	def partial_fit(self, X, y, classes=None, sample_weight=None):
		X = check_array(X, accept_sparse='csr', dtype=np.float64)
		_, n_features = X.shape


        self.coef_ = self._get_coef()
        #self.intercept_ = self._get_intercept()

        if _check_partial_fit_first_call(self, classes):
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_effective_classes = len(classes) if len(classes) > 1 else 2
            self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
            self.feature_count_ = np.zeros((n_effective_classes, n_features), dtype=np.float64)
        elif n_features != self.coef_.shape[1]:
            msg = "Number of features %d does not match previous data %d."
            raise ValueError(msg % (n_features, self.coef_.shape[-1]))

        Y = label_binarize(y, classes=self.classes_)
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        n_samples, n_classes = Y.shape

        if X.shape[0] != Y.shape[0]:
            msg = "X.shape[0]=%d and y.shape[0]=%d are incompatible."
            raise ValueError(msg % (X.shape[0], y.shape[0]))

        # label_binarize() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently
        Y = Y.astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= check_array(sample_weight).T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        self._count(X, Y)

        # XXX: OPTIM: we could introduce a public finalization method to
        # be called by the user explicitly just once after several consecutive
        # calls to partial_fit and prior any call to predict[_[log_]proba]
        # to avoid computing the smooth log probas at each call to partial fit
        self._update_feature_log_prob()
        self._update_class_log_prior(class_prior=class_prior)
        return self

    def _update_class_log_prior(self, class_prior=None):
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError("Number of priors must match number of"
                                 " classes.")
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            # empirical prior, with sample_weight taken into account
            self.class_log_prior_ = (np.log(self.class_count_) - np.log(self.class_count_.sum()))
        else:
            self.class_log_prior_ = np.zeros(n_classes) - np.log(n_classes)

    def _get_coef(self):
        return (self.feature_log_prob_[1:]
                if len(self.classes_) == 2 else self.feature_log_prob_)

    def _get_intercept(self):
        return (self.class_log_prior_[1:]
                if len(self.classes_) == 2 else self.class_log_prior_)

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = (np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse='csr')
        return (safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_)

    def predict(self, X):

        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

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


"""
n = 14000

data=np.loadtxt('../../data/adience/dataset.csv',delimiter=',')
classes=np.loadtxt('../../data/adience/gender_label.csv',delimiter=',')

data_train = data[0:n,:]
data_test = data[n:,:]
classes_train = classes[0:n]
classes_test = classes[n:]
"""

data_train = np.array([[6,180,12],[5.92,190,11],[5.58,170,12],[5.92,165,10],[5,100,6],[5.5,150,8],[5.42,130,7],[5.75,150,9]])

data_test = np.array([[6,130,8],[5.92,190,11],[5.58,170,12]])

classes_train = np.array([1,
        1,
        1,
        1,
        0,
        0,
        0,
        0])

classes_test =np.array([0, 1, 1])


n_data=data_train.shape[0]
#w = np.ones((n_data),dtype=float)/n_data
w =np.array([[0.5, 1, 0.5, 1, 1, 0.5, 1, 1]])

clf = MultinomialNB(alpha=1, class_prior=None, fit_prior=False)
clf.fit(data_train, classes_train, sample_weight=w)

Y_predict_proba = clf.predict_proba(data_test)
#Y_hat = clf.predict(data_test)
print Y_predict_proba 

#print(classification_report(classes_test , Y_hat))

