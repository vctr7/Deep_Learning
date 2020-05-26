from keras.layers import Dense
from keras.models import Sequential

import time
import numpy as np

m = 1000
n = 100
K = 1000


# load
load_train = np.load('train_data.npz')
x_train = load_train['xtrain']
y_train = load_train['ytrain']

load_test = np.load('test_data.npz')
x_test = load_test['xtest']
y_test = load_test['ytest']

x_train = np.array(x_train).T
x_test = np.array(x_test).T
y_train = np.array(y_train)
y_test = np.array(y_test)


model = Sequential()
model.add(Dense(3, input_shape=(2, ), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

"""
model.compile
    optimizers = adam, SGD, RMSprop
    loss = binary_crossentropy, mean_squared_error
"""
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
start1 = time.time()

"""
model.fit
    batch_size = 1, 32(default), 128, 1000
"""
model.fit(x_train, y_train, epochs=K)
train_time = time.time() - start1

start2 = time.time()
test_loss, test_acc = model.evaluate(x_test, y_test)
test_time = time.time() - start2
print("testing time :", time.time() - start2)

train_loss, train_acc = model.evaluate(x_train, y_train)

print("Train ACCURACY :", str(train_acc*100) + '%')
print("Train TIME :", train_time)

print("Test ACCURACY :", str(test_acc*100) + '%')
print("Test TIME :", test_time)