import random
import math
import numpy as np
import time

x1, x2, y = [], [], []
n = 100
m = 100
K = 1000

alpha = 0.0001

w = np.zeros((2, 1))
b = 0


for i in range(1100):
    x1.append(random.randint(-10, 10))
    x2.append(random.randint(-10, 10))
    if x1[-1] + x2[-1] > 0:
        y.append(1)
    else:
        y.append(0)


x_train = np.array([x1[:m], x2[:m]])
y_train = np.array(y[:m])

x_test = np.array([x1[-n:], x2[-n:]])


# measure the time of vectorization and compare with for-loops
start = time.time()
for iter in range(K):
    z = np.dot(w.T, x_train) + b
    a = 1 / (1 + pow(math.e, -z))
    dz = a - y_train
    dw = np.dot(x_train, dz.T) / m
    db = np.sum(dz) / m

    # update
    w = w - alpha*dw
    b = b - alpha*db
print("time :", time.time() - start)

new_y = []
new_z = np.dot(w.T, x_train)
new_a = 1 / (1 + pow(math.e, -new_z))
y_test = y_train[-n:]
cnt = 0

for i in range(m):
    if new_a[0][i] > 0.5:
        new_y.append(1)
    else:
        new_y.append(0)

    if new_y[i] == y_train[i]:
        cnt += 1

print("train accuracy :", cnt*100 / m)

new_y = []
new_z = np.dot(w.T, x_test)
new_a = 1 / (1 + pow(math.e, -new_z))
cnt = 0
for i in range(n):
    if new_a[0][i] > 0.5:
        new_y.append(1)
    else:
        new_y.append(0)

    if new_y[i] == y_test[i]:
        cnt += 1
print("test accuracy :", cnt*100 / n)
