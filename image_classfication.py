#8/13/18 
#cnn classification 

#neural network transforms an input by putting it through a series of hidden layers 
#layers made-up of neurons-connected neurons to neurons 

#cnn-layers built in 3 dimensions (width,height,depth)
#for cnn, neurons in one layer don't connect to all neurons in the next layer (only a small fraction of other neurons)
#cnn-i.hidden layers/feature extraction ii. classification (probability)
import numpy as np 
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
from keras.models import Sequential

#filter/kernels produce a feature map
#relu-to make the output non-linear 
#stide-the size of the step that the convolutional filter moves each time (filter slides pixel by pixel)

#feed the image into the model (512 x 512 pixels with 3 channels)
img_shape=(28,28,1)

#hidden layers/feature extraction 
model=Sequential()
#add a convolutional layer with 3x3 by 3 filters and a stride of one 
#set padding so that the input size equals the output size
model.add(Conv2D(6,2,input_shape=img_shape))
#add a relu activation to the layer
model.add(Activation('relu'))
#Pooling
model.add(MaxPool2D(2))

#fully connected layers
#use flatten to convert 3d to 1d 
model.add(Flatten())
#add dense layer with ten neurons 
model.add(Dense(10))
#using the softmax activation function for the last layer 
model.add(Activation('softmax'))
#model summary 
model.summary 

#classification (optimizer, loss, and metrics) 
from keras.datasets import mnist 

(x_train,y_train),(x_test,y_test)=mnist.load_data() 
x_train=np.expand_dims(x_train,-1)
x_test=np.expand_dims(x_test,-1)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',
metrics=['acc'])

#train the model, iterating on the data in batches of 32 samples with 5 epochs 
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test,y_test))