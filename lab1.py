# Plot ad hoc mnist instances
from keras.datasets import mnist
import tensorflow as tf
from matplotlib import pyplot
from matplotlib import image
from os import listdir
#import opencv as cv
from PIL import Image
import PIL.ImageOps
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale




import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def delete_background(img, threshold):
	for i in range(0, img.shape[0]):
		for j in range(0, img.shape[1]):
			if img[i, j] < threshold:
				img[i, j] = 0

	return img


data_list = list()

def load_data():
	for filename in listdir("data"):
		img_data = Image.open("data/" + filename)
		img_resize = img_data.resize((28, 28))
		img_l = img_resize.convert(mode="L")
		img_inverse = PIL.ImageOps.invert(img_l)
		img_arr = numpy.array(img_inverse)
		img_back = delete_background(img_arr, 120)


		data_list.append(img_back)


		#print("> loaded %s %s %s" % (filename, img_arr.shape, img_arr.dtype))

	for i in range(0, 10):
		pyplot.subplot(5, 2, i + 1)
		pyplot.imshow(data_list[i], cmap=pyplot.get_cmap('gray'))



(X_train, y_train), (X_test, y_test) = mnist.load_data()



#pyplot.imshow(X_train[0], cmap=pyplot.get_cmap('gray'))
#pyplot.subplot(224)
#pyplot.imshow(X_train[1], cmap=pyplot.get_cmap('gray'))
# show the plot




#X_test[0].show()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
print(X_train.shape)
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255


# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


print("origin data:")
print(X_test[1].shape)
print(X_test[1].dtype)
print("my data:")


# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

load_data()
data_x = numpy.asarray(data_list)
data_y = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

num_pixels_my = data_x.shape[1] * data_x.shape[2]
data_x = data_x.reshape(data_x.shape[0], num_pixels_my).astype('float32')

data_x = data_x / 255

data_y = np_utils.to_categorical(data_y)
print("my model")
scores = model.evaluate(data_x, data_y)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
pyplot.show()




