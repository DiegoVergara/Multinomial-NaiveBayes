import numpy as np
import math
import random
#from scipy.special import gammaln
#from scipy.special import psi
#from itertools import combinations
from numpy.core.umath_tests import inner1d
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import roc_curve,auc,confusion_matrix, classification_report, precision_score

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
        #index=np.random.permutation(np.arange(self.n_data))
        #train_test_split=0.8
        #self.idx_train=index[:np.int(self.n_data*train_test_split)]
        #self.idx_test=index[np.int(self.n_data*train_test_split):]

    def samme_proba(self, estimator):
        """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].
        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.
        """
        proba = estimator.predict_proba(self.data_test)

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = np.log(proba)

        return (self.n_classes - 1) * (log_proba - (1. / self.n_classes) * log_proba.sum(axis=1)[:, np.newaxis])
       
        
    def boost(self,iteration, w):
        clf = GaussianNB()
        clf.fit(self.data, self.labels, sample_weight=w)
        #######
        Y_predict_proba = clf.predict_proba(self.data)
        print Y_predict_proba 
        
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
        print Y_predict_proba;

        alpha = (-1. * self.learning_rate * (((self.n_classes -1.) / self.n_classes) * inner1d(y_coding, np.log(Y_predict_proba))))

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



data_train=np.loadtxt('gender_dataset_train.csv',delimiter=',')
classes_train=np.loadtxt('gender_dataset_train_label.csv',delimiter=',')
data_test=np.loadtxt('gender_dataset_test.csv',delimiter=',')
classes_test=np.loadtxt('gender_dataset_test_label.csv',delimiter=',')

n_estimators = 2
learning_rate = 1.0
alpha = 1.0


ada_mnb=adaboost_mnb(data_train, classes_train, data_test, n_estimators, learning_rate, alpha)
ada_mnb.fit()
Y_hat = ada_mnb.predict()
print(classification_report(classes_test , Y_hat))



