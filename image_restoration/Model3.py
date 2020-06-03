from keras.datasets import cifar10
from keras.models import Input, Model
from keras.layers import Conv2D, GaussianNoise, add, BatchNormalization, ReLU
import numpy as np
import cv2

# Model configuration
img_width, img_height = 32, 32
batch_size = 32
no_epochs = 100

# Open noised file
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

# Create model
input_img = Input(shape=(None, None, 3))
y = GaussianNoise(0.1, input_shape=(None, None, 3))(input_img)
initial_input = y

# 1st
y = Conv2D(64, 3, padding='same')(y)
y = BatchNormalization()(y)
y = ReLU()(y)
# 2nd
y = Conv2D(64, 3, padding='same')(y)
y = BatchNormalization()(y)
y = ReLU()(y)
# 3rd
y = Conv2D(64, 3, padding='same')(y)
y = BatchNormalization()(y)
y = ReLU()(y)
# 4th
y = Conv2D(64, 3, adding='same')(y)
y = BatchNormalization()(y)
y = ReLU()(y)
# 5th
y = Conv2D(3, 3, padding='same')(y)
# skip connection
y = add([y, initial_input])

model = Model([input_img], y)
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=no_epochs, batch_size=batch_size)

# Generate denoised images
denoised_image = model.predict(original)
denoised_image = np.reshape(denoised_image, (512, 512, 3))
denoised_image = denoised_image * 255
denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR)

# Save file
cv2.imwrite('model3.png', denoised_image)
