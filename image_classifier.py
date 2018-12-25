#image classifier 

import numpy as np 
import pandas as pd 
import keras 
from keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(
    rotation_range=40, #randomly roates the pictures 
    width_shift_range=0.2, #randomly move/translate the pics horizontally and vertically 
    height_shift_range=0.2,
    rescale=1./255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True,
    fill_mode='nearest' #filling newly created pixels 
)

from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array,load_img 

img=load_img('/Users/jmc/10815824_2997e03d76.jpg') #image 
img1=load_img('/Users/jmc/Downloads/Flicker8k_Dataset/17273391_55cfc7d3d4.jpg')
img11=load_img('/Users/jmc/Downloads/Flicker8k_Dataset')
img2=os.path.expanduser('~/Downloads/Flicker8k_Dataset')
x=img_to_array(img1) #np array (3, 150, 150)
x = x.reshape((1,) + x.shape)  

i=0 
for batch in datagen.flow(x,batch_size=1,save_to_dir='cat.4002.jpg'):
    i +=1
    if i>20:
        break 


#weight regularization 
#dropout prevents the nn from seeing the same pattern twice 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout,Flatten,Dense 

model = Sequential()
model.add(Conv2D(32,(3, 3), input_shape=(3, 150,150))) 
model.add(Activation('relu'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",optimizer='rmsprop',metrics=['accuracy'])
batch_size=16

train_datagen=ImageDataGenerator(rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1/.255)

train_generator=train_datagen.flow_from_directory(
    '/Users/jmc/Downloads/Flicker8k_Dataset/17273391_55cfc7d3d4.jpg', 
    target_size=(150,150),
    batch_size=batch_size,
    class_mode='binary') # binary crossentropy 

validation_generator=test_datagen.flow_from_directory(
    '/Users/jmc/Downloads/',
    target_size=(150,150),
    batch_size=batch_size,
    class_mode='binary'
)

model.fit_generator(
    train_generator,
    steps_per_epoch=5,
    validation_data=validation_generator,
    validation_steps=4
)

model.save_weights('first_taste.h5')