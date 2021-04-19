# Plot ad hoc mnist instances
from keras.datasets import mnist

from matplotlib import pyplot
from matplotlib import image
from os import listdir
#import opencv as cv
from PIL import Image
import PIL.ImageOps
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
#plt.subplot(221)
#plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
#plt.subplot(222)
#plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
#plt.subplot(223)
#plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
#plt.subplot(224)
#plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
#plt.show()



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

data_list = list()

def load_data():
	i = 0
	for filename in listdir("data"):
		img_data = Image.open("data/" + filename)
		img_resize = img_data.resize((28, 28))
		img_l = img_resize.convert(mode="L")
		img_inverse = PIL.ImageOps.invert(img_l)
		img_arr = numpy.array(img_inverse)

		print("> loaded %s %s %s" % (filename, img_arr.shape, img_arr.dtype))
		data_list.append(img_resize)
		#data_list[i].show()
		#i = i + 1
	img_l.show()
	img_inverse.show()

	#data = Image.open('data/image_0.jpg')
	#data = image.imread('data/image_0.jpg')

	#print()
	#print(data.size)
	#data_resize = data.resize((28, 28))

	#print(data_resize.size)
	#data_resize.show()





# load data

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("origin data:")
print(X_test[1].shape)
print(X_test[1].dtype)
print("my data:")
load_data()
# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

