import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve,auc,confusion_matrix, classification_report
from sklearn import datasets
from random import randint
from sklearn.ensemble import AdaBoostClassifier


data=np.loadtxt('../../data/adience/dataset.csv',delimiter=',')
classes=np.loadtxt('../../data/adience/gender_label.csv',delimiter=',')

n = 14000
n_estimators = 100

data_train = data[0:n,:]
data_test = data[n:,:]
classes_train = classes[0:n]
classes_test = classes[n:]

print 'init 2'
clf = AdaBoostClassifier(MultinomialNB(alpha=1, class_prior=None, fit_prior=False), algorithm = 'SAMME.R', n_estimators = n_estimators)
#clf = AdaBoostClassifier(GaussianNB(), algorithm = 'SAMME.R', n_estimators = n_estimators)

#clf = MultinomialNB(alpha=1, class_prior=None, fit_prior=False)
#clf = GaussianNB()
n_data=data_train.shape[0]
#w = np.ones((n_data),dtype=float)/n_data
#clf.fit(data_train, classes_train, sample_weight=w)
#print clf.predict_proba(data_test)
clf.fit(data_train, classes_train)
Y_hat = clf.predict(data_test)

print Y_hat
print classes_test

print(classification_report(classes_test , Y_hat))