# Logistic Regression learning by gradient ascent #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

max_iteration = 300
lamb = 0.001

def logistic_regression(X_train, Y_train, X_test, Y_test, eta, lamb, error, max_iteration):
    N = X_train.shape[0]
    ### Normalize the variables of the training set and test set ###
    mu = np.mean(X_train, 0)
    X_train = X_train - mu
    X_test = X_test - mu
    sigma = np.std(X_train, 0)
    usecol = sigma!= 0
    X_train = X_train[:, usecol]/sigma[usecol]
    X_test = X_test[:, usecol]/sigma[usecol]
    ### Add one to the first column of training set and test set ###
    X_train = np.hstack((np.ones((N, 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    ### Change labels to 0 and 1 ###
    Y_train[Y_train == -1] = 0
    Y_train = Y_train.reshape((-1, 1))
    Y_test[Y_test == -1] = 0
    Y_test = Y_test.reshape((-1, 1))
    
    ### log-likelihood function ###
    def log_likelihood(x, y, w):
        return -np.sum(y*x.dot(w) - np.log(1 + np.exp(x.dot(w))))

    ### Gradient of log-likelihood function ###
    def gradient_log_likelihood(x, y, w):
        return x.T.dot(y - 1/(1 + np.exp(-x.dot(w))))

    W = np.zeros((X_train.shape[1], 1))
    loglikelihood = log_likelihood(X_train, Y_train, W)
    number_iteration = 1
    delta = 1

    while (number_iteration <= max_iteration) & (delta > error):
        W = (1 - eta*lamb)*W + eta/N*gradient_log_likelihood(X_train, Y_train, W)
        loglikelihood = np.hstack((loglikelihood, log_likelihood(X_train, Y_train, W)))
        delta = abs((loglikelihood[number_iteration] - loglikelihood[number_iteration - 1])/loglikelihood[number_iteration - 1])
        number_iteration += 1

    ### Calculate misclassification error ###
    def log_reg(x, w):
        return np.array(1/(1 + np.exp(-x.dot(w)))>0.5, dtype='float64')

    error_train = np.mean(log_reg(X_train, W)!=Y_train)
    error_test = np.mean(log_reg(X_test, W)!=Y_test)

    ### Generate table ###
    out = pd.DataFrame(np.hstack((eta, number_iteration - 1, error_train, error_test, loglikelihood[number_iteration - 1])).reshape((1, -1)))
    out.columns = ['eta', 'Number of iteration', 'Error of train set', 'Error of test set', 'Negative Log-Likelihood']
    out.index = np.repeat('', out.shape[0])
    print(out.T)

    ### Plot ###
    plt.figure()
    plt.title('Negative Loglikelihood VS. No of iteration');
    plt.xlabel('Number of iterations');
    plt.ylabel('Negative Log-likelihood');
    plt.plot(range(0, number_iteration), loglikelihood);
    plt.show();
    return out;

### Question a ###
X_train_g = np.genfromtxt(r'gisette_train.data')
Y_train_g = np.genfromtxt(r'gisette_train.labels')
X_test_g = np.genfromtxt(r'gisette_valid.data')
Y_test_g = np.genfromtxt(r'gisette_valid.labels')
out_g = logistic_regression(X_train_g, Y_train_g, X_test_g, Y_test_g, 0.1, lamb, 0.001, max_iteration)

### Question b ###
X_train_hv = np.genfromtxt(r'X.dat')
Y_train_hv = np.genfromtxt(r'Y.dat')
X_test_hv = np.genfromtxt(r'Xtest.dat')
Y_test_hv = np.genfromtxt(r'Ytest.dat')
out_hv = logistic_regression(X_train_hv, Y_train_hv, X_test_hv, Y_test_hv, 0.08, lamb, 0.00001, max_iteration)

### Question c ###
X_train_d = pd.read_csv('dexter_train.csv', header = None)
Y_train_d = np.genfromtxt(r'dexter_train.labels')
X_test_d = pd.read_csv('dexter_valid.csv', header = None)
Y_test_d = np.genfromtxt(r'dexter_valid.labels')
X_train_d = np.array(X_train_d)
X_test_d = np.array(X_test_d)
out_d = logistic_regression(X_train_d, Y_train_d, X_test_d, Y_test_d, 0.001, lamb, 0.0001, max_iteration)
