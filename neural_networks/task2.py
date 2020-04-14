import numpy as np
import time

n = 100
m = 1000
K = 1000
alpha = 0.4


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy(y, y_hat):
    return -(np.dot(y.T, np.log(np.maximum(1e-10, y_hat)).T) + np.dot((1-y.T), np.log(np.maximum(1e-10, 1 - y_hat.T))))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


# load
load_train = np.load('train_data.npz')
x_train = load_train['xtrain']
y_train = load_train['ytrain']

load_test = np.load('test_data.npz')
x_test = load_test['xtest']
y_test = load_test['ytest']

# initialization
w1 = np.random.randn(2, 1) / np.sqrt(2)
w2 = np.random.randn(1, 1) * 0.01
b1 = np.random.randn(1, 1) / np.sqrt(2)
b2 = np.random.randn(1, 1) * 0.01

# train
start = time.time()
for iter2 in range(K):

    # 1st layer
    z1 = np.dot(w1.T, x_train) + b1
    a1 = tanh(z1)

    # 2nd layer
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    # back propagation of 2nd layer
    dz2 = a2 - y_train
    dw2 = np.dot(a1, dz2.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    # back propagation of 1st layer
    dz1 = np.multiply(np.dot(w2, dz2), 1 - np.power(a1, 2))     # derive tanh(x) = (1 - x^2)
    dw1 = np.dot(x_train, dz1.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    # update
    w1 = w1 - (dw1*alpha)
    b1 = b1 - (db1*alpha)
    w2 = w2 - (dw2*alpha)
    b2 = b2 - (db2*alpha)

a2[a2 > 0.5] = 1
a2[a2 <= 0.5] = 0
acc1 = np.sum(a2 == y_train)
cost = cross_entropy(y_train, a2) / m
checkpoint1 = time.time()

print("=============Train==============")
print('ACCURACY :' + str((100*acc1)/m) + '%')
print('TIME : ', checkpoint1 - start)
print('COST :', cost)

# test
new_z1 = np.dot(w1.T, x_test) + b1
new_a1 = tanh(new_z1)
new_z2 = np.dot(w2, new_a1) + b2
new_a2 = sigmoid(new_z2)

new_a2[new_a2 > 0.5] = 1
new_a2[new_a2 <= 0.5] = 0
acc2 = np.sum(new_a2 == y_test)
cost = cross_entropy(y_test, new_a2) / n

print("=============Test==============")
print('ACCURACY :' + str((100*acc2)/n) + '%')
print('TIME : ', time.time() - checkpoint1)
print('COST :', cost)
