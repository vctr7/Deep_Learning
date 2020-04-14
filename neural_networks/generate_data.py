import numpy as np
import random

m = 1000
n = 100


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


# save
x_train, y_train = generate(m)
np.savez('train_data', xtrain=x_train, ytrain=y_train)

x_test, y_test = generate(n)
np.savez('test_data', xtest=x_test, ytest=y_test)
