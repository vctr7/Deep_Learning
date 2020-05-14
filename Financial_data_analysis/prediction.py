import numpy as np
import pandas as pd
import keras as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def norm(data):
    normalized = []
    data = pd.to_numeric(data)
    for i in range(len(data)):
        normal = (data[i] - np.mean(data)) / (np.std(data))
        normalized.append(normal)

    return normalized


def generate_y(data):
    y = []
    for elem in data:
        if elem[0] <= -5 and elem[1] < 0:
            y.append(0)
        elif -5 < elem[0] <= -3 and elem[1] < 0:
            y.append(1)
        elif 5 > elem[0] >= 3 and elem[1] > 0:
            y.append(3)
        elif elem[0] >= 5 and elem[1] > 0:
            y.append(4)
        else:
            y.append(2)

    return y


df = pd.read_csv('IXIC.csv')

x_data = df.values[:, [5, 6]]
date = pd.to_datetime(df["Date"])

one_m, thr_m = [], []
for x in range(x_data.shape[0]):
    one_m.append(x_data[x][0])
    thr_m.append(x_data[x][1])

y_data = K.utils.to_categorical(np.array(generate_y(x_data)).astype('float32'))
x_data = np.array([norm(one_m), norm(thr_m)]).T.astype('float32')
x_train, x_test, y_train, y_test, _, date = train_test_split(x_data, y_data, date, test_size=0.8, shuffle=False)


# make Model
model = K.models.Sequential()
model.add(K.layers.Dense(128, input_shape=(2, )))
model.add(K.layers.LeakyReLU(alpha=0.01))
model.add(K.layers.Dense(256))
model.add(K.layers.LeakyReLU(alpha=0.01))
model.add(K.layers.Dense(128))
model.add(K.layers.LeakyReLU(alpha=0.01))
model.add(K.layers.Dense(5, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=500, batch_size=128)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Accuracy :", test_acc*100)

predicted = np.argmax(model.predict(x_test), axis=1)
real = np.argmax(y_test, axis=1)

# plot result
plt.plot(date, predicted, label='predicted')
plt.plot(date, real, label='real')
plt.legend()
plt.show()

