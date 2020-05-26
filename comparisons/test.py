from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('creditcard.csv')
df = df.drop(['Time'], axis=1)
# df.head()

frauds = df.loc[df['Class'] == 1]
non_frauds = df.loc[df['Class'] == 0]

print("We have", len(frauds), "fraud data points and", len(non_frauds), "regular data points.")

df_x = df.iloc[:, 0:29]
df_y = df[['Class']]

print("Size of All set :", df_x.shape, df_y.shape)

scaler = MinMaxScaler()
df_x = scaler.fit_transform(df_x)

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33)


#build model
model =Sequential()
model.add(Dense(29, input_dim=x_train.shape[1], activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size=128)

scores = model.evaluate(x_test, y_test)