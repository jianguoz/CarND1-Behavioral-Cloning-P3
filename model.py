import csv
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt


samples=[]
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)
del(samples[0])

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
	num_samples = len(samples)
	while True: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples=samples[offset: offset+batch_size]

			# images = []
			# measurements = []
			augmented_images, augmented_measurements=[], []
			for line in batch_samples:
				# Create adjusted steering measurements for the side camera images
				correction = float(0.25) #this is a parameter to tune

				source_path = line[0]
				steering = float(line[3])

				camera = np.random.choice(['center', 'left', 'right'])
				if(camera == 'left'):
					source_path = line[1]
					steering += correction
				elif(camera == 'right'):
					source_path = line[2]
					steering -= correction
				
				filename = source_path.split('/')[-1]
				current_path = './data/IMG/' + filename
				image = cv2.imread(current_path)
			
				# decide whether to horizontally flip the image
				flip_prob = np.random.random()
				if(flip_prob > 0.5):
					# flip the image and reverse the steering angle
					steering = -1.0*steering
					image = cv2.flip(image, 1)
				
				augmented_images.append(image)
				augmented_measurements.append(steering)	
			
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements)

			yield (X_train, y_train)


# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

import gc
from keras import backend as K
from keras.models import Sequential
from keras.layers import Lambda, Dropout, Cropping2D, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu')) #default stride (1,1), non-strided convoution
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)


model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=2, verbose=1)

# gc.collect()
# K.clear_session()
# history_object = model.fit_generator(train_generator, samples_per_epoch =
# 	len(train_samples), validation_data = 
# 	validation_generator,
# 	nb_val_samples = len(validation_samples), 
# 	nb_epoch=5, verbose=1)

# ### print the keys contained in the history object
# print(history_object.history.keys())

# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

model.save('model.h5') 


# def generator(samples, batch_size=32):
# 	num_samples = len(samples)
# 	while 1: # Loop forever so the generator never terminates
# 		shuffle(samples)
# 		for offset in range(0, num_samples, batch_size):
# 			batch_samples=samples[offset: offset+batch_size]

# 			images = []
# 			measurements = []
# 			for line in batch_samples:
# 				# Create adjusted steering measurements for the side camera images
# 				correction = 0.23 #this is a parameter to tune
# 				steering_center = float(line[3])
# 				for i in range(3):
# 					source_path = line[i]
# 					filename = source_path.split('/')[-1]
# 					current_path = './data/IMG/' + filename
# 					image = cv2.imread(current_path)
# 					images.append(image)
# 					#Currently only consider steering as labels
# 					if(i==0): #Center
# 						measurements.append(steering_center)
# 					elif(i==1): #Left
# 						steering_left = steering_center + correction
# 						measurements.append(steering_left)
# 					else: #Right
# 						steering_right = steering_center - correction
# 						measurements.append(steering_right)
			
# 			augmented_images, augmented_measurements=[], []
# 			for image, measurement in zip(images, measurements):
# 				augmented_images.append(image)
# 				augmented_measurements.append(measurement)
# 				#image_flipped = np.fliplr(image)
# 				image_flipped=cv2.flip(image, 1)
# 				measurement_flipped = -measurement
# 				augmented_images.append(image_flipped)
# 				augmented_measurements.append(measurement_flipped)    


# 			X_train = np.array(augmented_images)
# 			y_train = np.array(augmented_measurements)

# 			yield shuffle(X_train, y_train)

