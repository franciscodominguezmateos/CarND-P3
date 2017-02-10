import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import math
from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import model_from_json

#prepare data
with open('data/driving_log.csv') as f:
	content=f.readlines()
l=[]
for s in content:
	l.append(s.split(','))
print("len(l)",len(l))
print(l[1])
print(l[2])

model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape=(80, 60, 3)))
model.add(MaxPooling2D((2,2)))
model.add((Dropout(0.5)))
model.add(Conv2D(16, 3, 3))
model.add(MaxPooling2D((2,2)))
model.add((Dropout(0.5)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()
# TODO: Compile and train the model here.
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train, Y_train,
                    batch_size=128, nb_epoch=50,
                    verbose=1, validation_data=(X_val, Y_val))

def generate_arrays_from_file(path):
    while True:
        f = open(path)
        for line in f:
            # create numpy arrays of input data
            # and labels, from each line in the file
            x1, x2, y = process_line(line)
            yield ({'input_1': x1, 'input_2': x2}, {'output': y})
        f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
        samples_per_epoch=10000, nb_epoch=10)

model.save_weights('model.hd5')
json_string = model.to_json()
with open('model.json', 'w') as jfile:
    jfile.write(json_string)


