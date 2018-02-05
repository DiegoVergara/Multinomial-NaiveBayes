import numpy as np

X=np.loadtxt('../../data/adience/dataset.csv',delimiter=',')
y=np.loadtxt('../../data/adience/gender_label.csv',delimiter=',')
xc_sufficient = np.zeros((2,X.shape[1]))
sufficient = np.zeros((2,X.shape[1]))
theta  = np.zeros((2,X.shape[1]))
prior = np.zeros((2,1))
alpha = 1.0
for i in xrange(0,X.shape[0]):
	xc_sufficient[int(y[i])] += X[i,:]
	prior[int(y[i])] = (prior[int(y[i])]+1.0)/X.shape[0]
	sufficient[int(y[i])] += xc_sufficient[int(y[i])]
	theta[int(y[i])] = (sufficient[int(y[i])] + alpha) / (np.sum(sufficient[int(y[i])]) + X.shape[1]*alpha)

print "xc_sufficient"
print xc_sufficient

print "prior"
print prior


print "sufficient"
print sufficient
	
print "theta"
print theta