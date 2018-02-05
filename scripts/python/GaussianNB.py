import numpy as np
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot, logsumexp
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.fixes import in1d
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import roc_curve,auc,confusion_matrix,classification_report

class GaussianNB():

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]


    def predict_log_proba(self, X):

        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = np.log(np.sum(np.exp(jll),axis=1))
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        #return np.exp(self.predict_log_proba(X))
        #return (self.predict_log_proba(X))
        jll = self._joint_log_likelihood(X)
        max_value = np.amax(jll)
        min_value = np.amin(jll)
        jll = (jll-min_value)/(max_value-min_value)
        return np.exp(jll)

    def _update_mean_variance(self, n_past, mu, var, X, sample_weight=None):
        if X.shape[0] == 0:
            return mu, var

        # Compute (potentially weighted) mean and variance of new datapoints
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            new_mu = np.average(X, axis=0, weights=sample_weight / n_new)
            new_var = np.average((X - new_mu) ** 2, axis=0, weights=sample_weight / n_new)
        else:
            n_new = X.shape[0]
            new_var = np.var(X, axis=0)
            new_mu = np.mean(X, axis=0)

        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)

        # Combine mean of old and new data, taking into consideration
        # (weighted) number of observations
        total_mu = (n_new * new_mu + n_past * mu) / n_total

        # Combine variance of old and new data, taking into consideration
        # (weighted) number of observations. This is achieved by combining
        # the sum-of-squared-differences (ssd)
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = (old_ssd + new_ssd + (n_past / float(n_new * n_total)) * (n_new * mu - n_new * new_mu) ** 2)
        total_var = total_ssd / n_total

        return total_mu, total_var



    def fit(self, X, y, sample_weight=None):

        X, y = check_X_y(X, y)
        return self._partial_fit(X, y, np.unique(y), _refit=True, sample_weight=sample_weight)



    def _partial_fit(self, X, y, classes=None, _refit=False,sample_weight=None):
        X, y = check_X_y(X, y)

        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        epsilon = 1e-9 * np.var(X, axis=0).max()

        if _refit:
            self.classes_ = None

        if _check_partial_fit_first_call(self, classes):
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_features = X.shape[1]
            n_classes = len(self.classes_)
            self.theta_ = np.zeros((n_classes, n_features))
            self.sigma_ = np.zeros((n_classes, n_features))
            self.class_prior_ = np.zeros(n_classes)
            self.class_count_ = np.zeros(n_classes)
        else:
            if X.shape[1] != self.theta_.shape[1]:
                msg = "Number of features %d does not match previous data %d."
                raise ValueError(msg % (X.shape[1], self.theta_.shape[1]))
            # Put epsilon back in each time
            self.sigma_[:, :] -= epsilon

        classes = self.classes_

        unique_y = np.unique(y)
        unique_y_in_classes = in1d(unique_y, classes)

        if not np.all(unique_y_in_classes):
            raise ValueError("The target label(s) %s in y do not exist in the " "initial classes %s" %(y[~unique_y_in_classes], classes))

        for y_i in unique_y:
            i = classes.searchsorted(y_i)
            X_i = X[y == y_i, :]

            if sample_weight is not None:
                sw_i = sample_weight[y == y_i]
                N_i = sw_i.sum()
            else:
                sw_i = None
                N_i = X_i.shape[0]

            new_theta, new_sigma = self._update_mean_variance(self.class_count_[i], self.theta_[i, :], self.sigma_[i, :], X_i, sw_i)

            self.theta_[i, :] = new_theta
            self.sigma_[i, :] = new_sigma
            self.class_count_[i] += N_i

        self.sigma_[:, :] += epsilon
        self.class_prior_[:] = self.class_count_ / np.sum(self.class_count_)
        #print self.class_prior_[:]
        return self

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "classes_")

        X = check_array(X)
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) / (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood


n = 14000

data=np.loadtxt('../../data/adience/dataset.csv',delimiter=',')
classes=np.loadtxt('../../data/adience/gender_label.csv',delimiter=',')

data_train = data[0:n,:]
data_test = data[n:,:]
classes_train = classes[0:n]
classes_test = classes[n:]

'''
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
'''
n_data=data_train.shape[0]
w = np.ones((n_data),dtype=float)/n_data

clf = GaussianNB()
#clf.fit(data_train, classes_train, sample_weight=w)
clf.fit(data_train, classes_train)

#Y_predict_proba = clf.predict_proba(data_test)
#print Y_predict_proba 
Y_hat = clf.predict(data_test)
print(classification_report(classes_test , Y_hat))

