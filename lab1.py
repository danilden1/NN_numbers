import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow_datasets as tfds  # pip install tensorflow-datasets
import tensorflow as tf
import logging
import numpy as np
import time
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot

(trainX, trainy), (testX, testy) = mnist.load_data()
print("Train: X=%s, y=%s" % (trainX.shape, trainy.shape))
print("Test: X=%s, y=%s" % (testX.shape, testy.shape))

for i  in range(9):
   pyplot.subplot(330 + 1 + i)
   pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))

pyplot.show()
#tf.logging.set_verbosity(tf.logging.ERROR)
#tf.get_logger().setLevel(logging.ERROR)

def mnist_make_model(image_w: int, image_h: int):
   # Neural network model
   model = Sequential()
   model.add(Dense(784, activation='relu', input_shape=(image_w*image_h,)))
   model.add(Dense(10, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
   return model

def mnist_mlp_train(model):
   (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
   # x_train: 60000x28x28 array, x_test: 10000x28x28 array
   image_size = x_train.shape[1]
   train_data = x_train.reshape(x_train.shape[0], image_size*image_size)
   test_data = x_test.reshape(x_test.shape[0], image_size*image_size)
   train_data = train_data.astype('float32')
   test_data = test_data.astype('float32')
   train_data /= 255.0
   test_data /= 255.0
   # encode the labels - we have 10 output classes
   # 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0]
   num_classes = 10
   train_labels_cat = keras.utils.to_categorical(y_train, num_classes)
   test_labels_cat = keras.utils.to_categorical(y_test, num_classes)
   print("Training the network...")
   t_start = time.time()
   # Start training the network
   model.fit(train_data, train_labels_cat, epochs=8, batch_size=64,
             verbose=1, validation_data=(test_data, test_labels_cat))




model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

print(model.summary())

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])
model.fit(X, y, batch_size=1, epochs=1000, verbose=0)

print("Network test:")
print("XOR(0,0):", model.predict_proba(np.array([[0, 0]])))
print("XOR(0,1):", model.predict_proba(np.array([[0, 1]])))
print("XOR(1,0):", model.predict_proba(np.array([[1, 0]])))
print("XOR(1,1):", model.predict_proba(np.array([[1, 1]])))

## Parameters layer 1
W1 = model.get_weights()[0]
b1 = model.get_weights()[1]
## Parameters layer 2
W2 = model.get_weights()[2]
b2 = model.get_weights()[3]

print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)
print()

