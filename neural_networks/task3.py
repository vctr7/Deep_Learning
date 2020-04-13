import numpy as np
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy(y, y_hat):
    p1 = np.dot(y.T, np.log(y_hat).T)
    p2 = np.dot((1-y.T), np.log(1 - y_hat.T))
    return -(p1 + p2)


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


n = 100
m = 1000
K = 1000
alpha = 0.4

load_train = np.load('train_data.npz')
x_train = load_train['xtrain']
y_train = load_train['ytrain']

load_test = np.load('test_data.npz')
x_test = load_test['xtest']
y_test = load_test['ytest']

w1 = np.random.randn(3, 2) * 0.01
b1 = np.random.randn(3, 1) * 0.01
w2 = np.random.randn(1, 3) * 0.01
b2 = np.random.randn(1) * 0.01

# train
start = time.time()
for iter3 in range(K):
    # lst layer
    z1 = np.dot(w1, x_train) + b1
    a1 = tanh(z1)

    # 2nd layer
    z2 = np.dot(w2, a1) + b2
    aa = sigmoid(z2)
    cost = cross_entropy(y_train, aa)

    # back propagation
    dz2 = aa - y_train
    dw2 = np.dot(a1, dz2.T) / m
    db2 = np.sum(dz2) / m

    # derive tanh
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = np.dot(x_train, dz1.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    # update
    w1 = w1 - (dw1.T*alpha)
    b1 = b1 - (db1*alpha)
    w2 = w2 - (dw2.T*alpha)
    b2 = b2 - (db2*alpha)

aa[aa > 0.5] = 1
aa[aa <= 0.5] = 0
acc = np.sum(aa == y_train)
checkpoint1 = time.time()
print('train accuracy :' + str((acc*100)/m) + '%')
print('tratin time : ', checkpoint1 - start)

# test
new_z1 = np.dot(w1, x_test) + b1
new_a1 = tanh(new_z1)
new_z1 = np.dot(w2, new_a1) + b2
new_a2 = sigmoid(new_z1)
new_a2[new_a2 > 0.5] = 1
new_a2[new_a2 <= 0.5] = 0

acc = np.sum(new_a2 == y_test)
print('test accuracy :' + str((100*acc)/n) + '%')
print('test time : ', time.time() - checkpoint1)