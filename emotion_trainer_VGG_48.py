# CNN Trainer on Kaggle Challenge -- Challenges in Representation Learning: Facial Expression Recognition Challenge
# Author: Tarek Khattab
# Date: 03/1/2013


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
import matplotlib.pyplot as plt
import cv2

# Load the datasets
data = pd.read_csv('data/fer2013.csv', delimiter=',')

# Taking Training and PublicTest data for training and PrivateTest data for testing
Nbr_Train=32298
data_train = data[:Nbr_Train]
data_test = data[Nbr_Train:]
# data_train.head()
#data_train = data[:5]
#data_test = data[5:8]q

y_train = data_train['emotion'].values
y_test = data_test['emotion'].values

print(y_train.shape)
print(y_test.shape)

# Converting string of pixel data to an array
x_train = np.zeros((y_train.shape[0], 48*48))
for i in range(y_train.shape[0]):
    x_train[i] = np.fromstring(data_train['pixels'][i], dtype=int, sep=' ')

x_test = np.zeros((y_test.shape[0], 48*48))
for i in range(y_test.shape[0]):
#    x_test[i] = np.fromstring(data_test['pixels'][32298+i], dtype=int, sep=' ')
    x_test[i] = np.fromstring(data_test['pixels'][Nbr_Train+i], dtype=int, sep=' ')

print(x_train.shape)
print(x_test.shape)

# Generate reversed images for every data assuming emotion are symetric
img_rows, img_cols= 48, 48
num_classes = 7

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1 )
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

#Replicate the image
## Resize The images :
img_rows, img_cols, channels = 48, 48,3

#_train = np.zeros((y_train.shape[0],img_rows1, img_cols1, channels))
X_train = np.zeros((y_train.shape[0],img_rows, img_cols, channels))
x_train_rev = np.zeros((y_train.shape[0],img_rows, img_cols, channels))

for i in range(y_train.shape[0]):
    image=np.dstack((x_train[i],x_train[i],x_train[i]))
#    image=cv2.resize(image,(224,224))
    X_train[i]=image
    image=cv2.flip(image,0)
    x_train_rev[i]=image

X_test = np.zeros((y_train.shape[0],img_rows, img_cols, channels))
x_test_rev= np.zeros((y_test.shape[0],img_rows, img_cols, channels))
for i in range(y_test.shape[0]):
    image=np.dstack((x_test[i],x_test[i],x_test[i]))
#    image=cv2.resize(image,(224,224))
    X_test[i]=image
    image=cv2.flip(image,0)
    x_test_rev[i]=image

x_train=X_train
x_test =X_test

#x_train_rev = np.flip(x_train, 2)
#x_test_rev = np.flip(x_test, 2)

plt.figure(1)
plt.imshow(x_train[0].reshape((48,48,3)))

plt.figure(2)
plt.imshow(x_train_rev[0].reshape((48,48,3)))

# Some preprocessing
input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train_rev = x_train_rev.astype('float32')
x_test_rev = x_test_rev.astype('float32')
x_train /= 255
x_test /= 255
x_train_rev /= 255
x_test_rev /= 255
print('x_train shape:', x_train.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)



from keras.applications.vgg16 import VGG16
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

def vgg_model16_48pixels(image_size=48):
    #Load the VGG model
    vgg_conv = VGG16(weights='imagenet', 
                     include_top=False, 
                     input_shape=(image_size, image_size, 3))    
    # Freeze the layers except the last 4 layers
#    for layer in vgg_conv.layers[:-1]:
#        layer.trainable = False
     
    # Check the trainable status of the individual layers
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)

    # Create the model
    model = models.Sequential()
    # Add the vgg convolutional base model
    model.add(vgg_conv)
    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(7, activation='softmax'))
     
    # Show a summary of the model. Check the number of trainable parameters
    model.summary()    
    # Compile the model
#    model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])    
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.adam(),metrics=['accuracy'])
    
    return model

#def vgg_model():
#
#    model = VGG16()
#    model.compile(loss=keras.losses.categorical_crossentropy,
#                  optimizer=keras.optimizers.adam(),
#                  metrics=['accuracy'])
#    
#    print(model.summary())
#    
#    return 

## define the model
#def cnn_model():
#    model = Sequential()
#
#    model.add(BatchNormalization(input_shape=input_shape))
#
#    model.add(Conv2D(32, kernel_size=(3, 3)))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#
#    model.add(Conv2D(64, (3, 3)))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
#
#    model.add(Conv2D(512, (3, 3)))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
#
#    model.add(Conv2D(64, (3, 3)))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    model.add(Dropout(0.25))
#    # model.add(Conv2D(32, (3, 3)))
#    # model.add(BatchNormalization())
#    # model.add(Activation('relu'))
#    # model.add(Dropout(0.2))
#
#    model.add(Flatten())
#    model.add(Dense(512))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    model.add(Dropout(0.25))
#
#    model.add(Dense(128))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    model.add(Dropout(0.25))
#
#    model.add(Dense(num_classes, activation='softmax'))
#
#    model.compile(loss=keras.losses.categorical_crossentropy,
#                  optimizer=keras.optimizers.adam(),
#                  metrics=['accuracy'])
#
#    return model

# function to plot graph
def plotGraph(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()

# We will train two models one on normal images another on reversed images and finally a NN on predicted values from these models

batch_size = 64
epochs = 100
model = []
training=0
PathtoModels="saved_model_vgg16"
import os 

if training :
    print("=======| Model 1 |=========")
    #modelc = cnn_model()
    #modelc =vgg_model()
    modelc =vgg_model16_48pixels(image_size=48)
    history = modelc.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split = 0.1)
    model.append(modelc)
    plotGraph(history)
    
    print("=======| Model 2 |=========")
    #modelc = cnn_model()
    #modelc =vgg_model()
    modelc =vgg_model16_48pixels(image_size=48)
    history = modelc.fit(x_train_rev, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split = 0.1)
    model.append(modelc)
    plotGraph(history)


    ## Create Directory
    
    PathtoModels="saved_model_vgg16"
    if not os.path.exists(PathtoModels):
        os.makedirs(PathtoModels)



## p_tr >> prediction on training data
## p_te >> prediction on test data
#
p_tr = []
p_te = []

for i, m in enumerate(model):
    if i ==0:
        p = m.predict(x_train)
        pt = m.predict(x_test)
    else:
        p = m.predict(x_train_rev)
        pt = m.predict(x_test_rev)
    p_tr.append(p)
    p_te.append(pt)
    m.save('saved_model_vgg16/cnn'+str(i)+'.h5')

modelc=vgg_model16_48pixels(image_size=48)
modelc.load_weights('saved_model_vgg16/cnn'+str(0)+'.h5')
model.append(modelc)

modelc=vgg_model16_48pixels(image_size=48)
modelc.load_weights('saved_model_vgg16/cnn'+str(1)+'.h5')
model.append(modelc)

print(len(model))

p_train = np.zeros((y_train.shape[0],num_classes*len(model)))
p_test = np.zeros((y_test.shape[0],num_classes*len(model)))

for i, p in enumerate(p_tr):
    print(i)
    p_train[:,num_classes*i:num_classes*(i+1)] = p

for i, p1 in enumerate(p_te):
    print(i)
    p_test[:,num_classes*i:num_classes*(i+1)] = p1

print(p_train.shape, p_test.shape)

# Trains an Conventional Neural Network on previously predicted values by the two models
batch_size = 32
num_classes = 7
epochs = 12

modele = Sequential()
modele.add(Dense(128, activation='relu', input_shape=(num_classes*len(model),)))
modele.add(Dense(num_classes, activation='softmax'))

modele.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = modele.fit(p_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(p_test, y_test))

score = modele.evaluate(p_test, y_test, verbose=0)
modele.save('saved_model_vgg16/ensemble.h5')

print('NN Based Ensembled Model')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
