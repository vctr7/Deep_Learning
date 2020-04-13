import random
import numpy as np
import time


def generate(num):
    x1, x2, y = [], [], []
    for i in range(num):
        x1.append(random.uniform(-2, 2))
        x2.append(random.uniform(-2, 2))

        if x1[-1] * x1[-1] > x2[-1]:
            y.append(1)
        else:
            y.append(0)

    return np.array((x1, x2)), np.array(y)


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

w = np.random.randn(2, 1) * 0.01
b = np.random.randn(1) * 0.01


x_train, y_train = generate(m)
np.savez('train_data', xtrain=x_train, ytrain=y_train)

x_test, y_test = generate(n)
np.savez('test_data', xtest=x_test, ytest=y_test)



# # measure the time of vectorization and compare with for-loops
start = time.time()
for iter in range(K):
    z = np.dot(w.T, x_train) + b
    a = sigmoid(z)
    # print(a)
    cost = cross_entropy(y_train, a) / m

    dz = a - y_train
    dw = np.dot(x_train, dz.T) / m
    db = np.sum(dz) / m
    # update
    w = w - (alpha*dw)
    b = b - (alpha*db)

    a[a > 0.5] = 1
    a[a <= 0.5] = 0
    acc = np.sum(a == y_train)

    # print output
    print('======Iteration ' + '%04d' % (iter + 1) + '=======')
    print('cost : ' + str(cost))
    print('w : ' + str(w))
    print('b : ' + str(b))

print('train accuracy :' + str((100*acc)/m) + '%')
checkpoint1 = time.time()
print("train time :", checkpoint1 - start)

# test the model
test_time = time.time()
new_y = []
new_z = np.dot(w.T, x_test) + b
new_a = sigmoid(new_z)
cost = cross_entropy(y_test, new_a) / n
new_a[new_a > 0.5] = 1
new_a[new_a <= 0.5] = 0
acc2 = np.sum(new_a == np.array(y_test))

print('test cost : ' + str(cost))
print('test accuracy : ' + str((100*acc2)/n))
print('test time : ', time.time()-test_time)
