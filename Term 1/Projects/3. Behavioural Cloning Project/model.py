import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg') # lets you use matplotlib on AWS where no display is available.
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []

with open('Data2/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None) # Skip the Excel File Headers
	for line in reader:
		lines.append(line)

# Randomly Split Data into 80% Training Data, and 20% Validation Data
training_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Created a Geneartor so that more images are loaded into Memmory as they are needed rather than all at once.
def generator(lines, batch_size=32):
	num_samples = len(lines)
	correction = 0.2 # correction factor for steering associated with left and right camera images
	while 1: # Loop forever so the generator never terminates
		shuffle(lines)
		for offset in range(0, num_samples, batch_size):
			batch_samples = lines[offset:offset+batch_size]
			images = []
			measurements = []
			for batch_sample in batch_samples:
                # Gets path to All 3 images
				for i in range(3):
					source_path = batch_sample[i]
					filename = source_path.split('\\')[-1]
					current_path = 'Data2/IMG/' + filename
					image = cv2.imread(current_path)
					images.append(image)

				# From the center cameras perspective, if it sees an image associated 
				# with the left camera, it needs to add a litle steering to the right to get back to center
				center_steering = float(batch_sample[3])
				left_steering = float(batch_sample[3]) + correction
				right_steering = float(batch_sample[3]) - correction
				measurements.append(center_steering)
				measurements.append(left_steering)
				measurements.append(right_steering)

			agumented_images, agumented_measurements = [], []
			for image, measurement in zip(images, measurements):
				agumented_images.append(image)
				agumented_measurements.append(measurement)
				agumented_images.append(cv2.flip(image,1))
				agumented_measurements.append(measurement*-1.0)

			X_train = np.array(images)
			y_train = np.array(measurements)
			yield shuffle(X_train, y_train)


# Process 32 lines at a time.
batch_size = 32
training_generator = generator(training_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D

model = Sequential()
# Normalize the input. Image values are made to have values from 0 to 1 and then they are mean centered by subtracted by 0.5 to make the mean 0

# a lambda layer in keras is for any arbitrary function that operators on the input of that layer
# Cropping2D is a layer in our model that crops images.
# top of the image has hills and trees, bottom has the cars hood.
# cropping them will help train the model faster
# it crops 70 pixels from the top and 25 from the bottom, 0 from the left and the right

# Nvidia Network Architecture (5 convolutional layers, and 4 Fully connected)
model.add(Cropping2D(cropping=((70,25), (0,0)),input_shape=(160,320,3)  ))
model.add(Lambda(lambda x: x/255.0 - 0.5))

# 24 filter layers makes the output depth 24, 5x5 kernerl for convolutions
# subsample is 2,2 meaning a stride of 2
# default is Valid Padding = (Width (or Height) – Filter Width + 1)/Strides.


# IN: 90x295x3 Out: 43x146x24
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))

# In: 43x146x24 Out: 20x71x36
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))

# In: 20x71x36 Out: 8x34x48
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))

#In: 8x34x48 Out: 6x32x64
model.add(Convolution2D(64,3,3,subsample=(1,1),activation='relu'))

# In: 6x32x64 Out: 4x30x64
model.add(Convolution2D(64,3,3,subsample=(1,1),activation='relu'))


model.add(Flatten()) 
model.add(Dense(100)) # a Fully Conected Layers
model.add(Dense(50))
model.add(Dense(10))

model.add(Dense(1)) # Output Layer


# No activation function (Softmax) because this is a not a 
# classification problem this requires a regression network (Get Continuous Values)

# not crossentropy error function due to it being a regression network
# verbose sets the outputs given after each epoch in the terminal
model.compile(loss='mse', optimizer='adam', verbose = 1)

# fit the model with data returned from the generators
historyOfModel = model.fit_generator(training_generator, samples_per_epoch= len(training_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')

### print the keys contained in the history object
print(historyOfModel.history.keys())

### Save a Plot of the training and validation loss for each epoch
plt.plot(historyOfModel.history['loss'])
plt.plot(historyOfModel.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('LossVisualization.jpg')
