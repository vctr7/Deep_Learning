from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, GaussianNoise
import numpy as np
import cv2

# Model configuration
img_width, img_height = 32, 32
batch_size = 32
no_epochs = 100

original = cv2.imread('noisy.png')
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
original = np.array(original)
original = original / 255.0
original = np.reshape(original, (-1, 512, 512, 3))

# Use cifar10 as train data
(input_train, target_train), (input_test, target_test) = cifar10.load_data()
input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 3)
input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 3)
input_shape = (img_width, img_height, 3)

# Normalize data
x_train = input_train.astype('float32') / 255
x_test = input_test.astype('float32') / 255
y_train = x_train
y_test = x_test

# Model
model = Sequential([GaussianNoise(0.1, input_shape=(None, None, 3))])
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(Conv2D(3, 3, padding='same'))

# Compile and fit data
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=no_epochs, batch_size=batch_size)

# Generate denoised image
denoised_image = model.predict(original)
denoised_image = np.reshape(denoised_image, (512, 512, 3))
denoised_image = denoised_image * 255
denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR)

# Save file
cv2.imwrite('model1.png', denoised_image)