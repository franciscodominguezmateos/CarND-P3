import os
import csv
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import sklearn
import math
from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation
from keras.layers import Dropout
from keras.layers import Lambda
from keras.utils import np_utils
from keras.models import model_from_json

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples.pop(0)# take away csv header
print(samples[0])
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Used as a reference pointer so code always loops back around
        #shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #center
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                #left
                #name = './data/IMG/'+batch_sample[1].split('/')[-1]
                #left_image = cv2.imread(name)
                #left_angle = float(batch_sample[3])+offset
                #images.append(left_image)
                #angles.append(left_angle)
                #right
                #name = './data/IMG/'+batch_sample[2].split('/')[-1]
                #right_image = cv2.imread(name)
                #right_angle = float(batch_sample[3])-offset
                #images.append(right_image)
                #angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images,np.float32)
            X_train = X_train[:,80:,:,:] 
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
t=next(train_generator)
print("t=",t[0].shape)
validation_generator = generator(validation_samples, batch_size=3)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=( row, col,ch),
        output_shape=(row, col, ch)))
model.add(Conv2D(32, 3, 3, input_shape=(row, col, ch)))
model.add(MaxPooling2D((2,2)))
#model.add((Dropout(0.5)))
model.add(Conv2D(16, 3, 3))
model.add(MaxPooling2D((2,2)))
#model.add((Dropout(0.5)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()
# TODO: Compile and train the model here.
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(train_generator, 
                    samples_per_epoch= len(train_samples), 
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), 
                    nb_epoch=3)

model.save_weights('model.hd5')
json_string = model.to_json()
with open('model.json', 'w') as jfile:
    jfile.write(json_string)


