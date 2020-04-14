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
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


# load
load_train = np.load('train_data.npz')
x_train = load_train['xtrain']
y_train = load_train['ytrain']

load_test = np.load('test_data.npz')
x_test = load_test['xtest']
y_test = load_test['ytest']

# initialization
w = np.random.randn(2, 1) / np.sqrt(2)
b = np.random.randn(1) / np.sqrt(2)

# train
start = time.time()
for iter in range(K):

    # forward
    z = np.dot(w.T, x_train) + b
    a = sigmoid(z)

    # backward
    dz = a - y_train
    dw = np.dot(x_train, dz.T) / m
    db = np.sum(dz) / m

    # update
    w = w - (alpha * dw)
    b = b - (alpha * db)

    a[a > 0.5] = 1
    a[a <= 0.5] = 0
    acc = np.sum(a == y_train)

    # print output
    # print('======Iteration ' + '%04d' % (iter + 1) + '=======')
    # print('cost : ' + str(cost))
    # print('w : ' + str(w))
    # print('b : ' + str(b))

cost = cross_entropy(y_train, a) / m
checkpoint1 = time.time()
print("=============Train==============")
print('ACCURACY :' + str((100 * acc) / m) + '%')
print('TIME : ', checkpoint1 - start)
print('COST :', cost)

# test
test_time = time.time()
new_y = []
new_z = np.dot(w.T, x_test) + b
new_a = sigmoid(new_z)

new_a[new_a > 0.5] = 1
new_a[new_a <= 0.5] = 0
acc2 = np.sum(new_a == np.array(y_test))
cost = cross_entropy(y_test, new_a) / n

print("=============Test==============")
print('ACCURACY :' + str((100 * acc2) / n) + '%')
print('TIME : ', time.time() - checkpoint1)
print('COST :', cost)
