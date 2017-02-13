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
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples.pop(0)# take away csv header
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# From: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.o92uic4yq
# Data aumentation with brightness, shadow and transformations
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
def trans_image(image,steer,trans_range):
    # Translation
    cols=image.shape[1]
    rows=image.shape[0]
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_tr,steer_ang
def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Used as a reference pointer so code always loops back around
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                side = np.random.randint(3)
                if side==0:
	            #center
                    name = './data/IMG/'+batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    image=center_image
                    angle=center_angle
                if side==1:
                    #left
                    name = './data/IMG/'+batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(name)
                    left_angle = float(batch_sample[3])+0.5
                    image=left_image
                    angle=left_angle
                if side==2:
                    #right
                    name = './data/IMG/'+batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(name)
                    right_angle = float(batch_sample[3])-0.5
                    image=right_image
                    angle=right_angle
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                #rows=image.shape[0]
                #cols=image.shape[1]
                flip=np.random.randint(5)
                if flip==1:
                    image = np.fliplr(image)
                    angle = -angle
                if flip==2:
                    image,angle=trans_image(image,angle,100)
                if flip==3:
                    image = augment_brightness_camera_images(image)
                if flip==4:
                    image = add_random_shadow(image)
                images.append(image)
                angles.append(angle)
            # trim image to only see section with road
            X_train = np.array(images,np.float32)
            X_train = X_train[:,40:-20,:,:] 
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32*8)
#Test generated images
t=next(train_generator)
for i in range(1):
    image=t[0][i]
    print("angle=",t[1][i])
    plt.figure()
    plt.imshow(image/255.0)
    plt.show()
validation_generator = generator(validation_samples, batch_size=32*8)

ch, row, col = 3, 160-(40+20), 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
# From NVIDIA paper: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
# In my case the input image is 120x320 not 66x220
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=( row, col,ch),
        output_shape=(row, col, ch)))
model.add(Conv2D(24, 5, 5, input_shape=(row, col, ch)))
model.add(MaxPooling2D((2,2)))
model.add((Dropout(0.5)))
model.add(Conv2D(36, 5, 5))
model.add(MaxPooling2D((2,2)))
model.add((Dropout(0.5)))
model.add(Conv2D(48, 3, 3))
model.add(MaxPooling2D((2,2)))
model.add((Dropout(0.5)))
model.add(Conv2D(64, 3, 3))
model.add(MaxPooling2D((2,2)))
model.add((Dropout(0.5)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()
# TODO: Compile and train the model here.
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
# checkpoint
filepath="model-checkpoint-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit_generator(train_generator, 
                    samples_per_epoch= len(train_samples)*3, 
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples),
                    callbacks=callbacks_list, 
                    nb_epoch=100)

model.save_weights('model.h5')
json_string = model.to_json()
with open('model.json', 'w') as jfile:
    jfile.write(json_string)


