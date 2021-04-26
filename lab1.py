# Plot ad hoc mnist instances
from keras.datasets import mnist
import tensorflow as tf
from matplotlib import pyplot
from matplotlib import image
from os import listdir
#import opencv as cv
from PIL import Image
from skimage.morphology import dilation
import PIL.ImageOps
from prettytable import PrettyTable
# load (downloaded if needed) the MNIST dataset
# plot 4 images as gray scale




import tensorflow.keras as keras

from keras import models
from keras import layers

# define baseline model
def baseline_model():
	# create model
	model = models.Sequential([
		layers.Conv2D(32, (3, 3), activation='relu',  # (3,3) - фильтр
					  input_shape=(28, 28, 1)),
		layers.MaxPooling2D((2, 2)),  # фильтр (2,2) для пулинга
		layers.Conv2D(64, (3, 3), activation='relu'),
		layers.MaxPooling2D((2, 2)),
		layers.Conv2D(64, (3, 3), activation='relu'),
		layers.Flatten(),
		layers.Dense(64, 'relu'),
		layers.Dense(10, 'softmax')
	])
	return model

def delete_background(img, threshold):
	for i in range(0, img.shape[0]):
		for j in range(0, img.shape[1]):
			if img[i, j] < threshold:
				img[i, j] = 0
			else:
				img[i, j] = 255

	return img


data_list = list()
import numpy
seed = 7
numpy.random.seed(seed)
def load_data():
	for filename in listdir("data"):
		img_data = Image.open("data/" + filename)
		img_resize = img_data.resize((28, 28))
		img_l = img_resize.convert(mode="L")
		img_inverse = PIL.ImageOps.invert(img_l)
		img_arr = numpy.array(img_inverse)
		img_back = delete_background(img_arr, 120)
		img_back = dilation(img_back)


		data_list.append(img_back)


		print("> loaded %s %s %s" % (filename, img_arr.shape, img_arr.dtype))

	for i in range(0, 10):
		pyplot.subplot(5, 2, i + 1)
		pyplot.imshow(data_list[i], cmap=pyplot.get_cmap('gray'))



(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(x_train[0, :, :])
plt.colorbar()

x_train = x_train.reshape((60000, 28, 28, 1))
x_train = x_train.astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1))
x_test = x_test.astype('float32') / 255

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from keras import models
from keras import layers
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', # (3,3) - фильтр
                        input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)), # фильтр (2,2) для пулинга
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, 'relu'),
    layers.Dense(10, 'softmax')
])

from datetime import datetime
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(x=x_train,
          y=y_train,
          epochs=1,
          batch_size=64,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])

scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

load_data()

data_x = numpy.asarray(data_list)
data_y = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

data_x = data_x.reshape((10, 28, 28, 1))
data_x = data_x.astype('float32') / 255

from keras.utils import np_utils
data_y = np_utils.to_categorical(data_y)
print("my model")
#scores = model.evaluate(data_x, data_y)

prediction = model.predict(data_x)

t = PrettyTable(['Numb', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
#print("Baseline Error: %.2f%%" % (100-scores[1]*100))
for i in range(0, 10):
	#print(i, ": ", end="")
	#for j in range(0, 10):
	#		print("%.2f%%" % (prediction[i, j] * 100), " ", end="")
	#print()
	pred = list()
	for j in range(0, 10):
		pred.append("%.2f%%" % (prediction[i,j] * 100))
	t.add_row([i, pred[0],
			   pred[1],
			   pred[2],
			   pred[3],
			   pred[4],
			   pred[5],
			   pred[6],
			   pred[7],
			   pred[8],
			   pred[9],])


print(t)

for i in range(0, 10):
	j = i
	if prediction[i, j] > 0.50:
		print(i, ": SUCCESS ", "%.2f%%" % (prediction[i, j] * 100))
	else:
		print(i, ": FUUUUCK ", "%.2f%%" % (prediction[i, j] * 100))



pyplot.show()





