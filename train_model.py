import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = Sequential()
#1st convolution layer
model.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#2nd convolution layer
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Flatten layer
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=8, activation='softmax')) # softmax for more than 2
model.add(Dropout(0.5))

#8 classes in total
#Compile model and obtain summary
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

##Defining training paramters
generate_train_data = ImageDataGenerator(rescale = 1./255,
                                        shear_range = 0.2,                                        zoom_range = 0.2,
                                        horizontal_flip = True)

train_dataset = generate_train_data.flow_from_directory(r'C:\Users\pa662\PycharmProjects\HGM\V11\datasets\train',
                                                 target_size = (64, 64),
                                                 batch_size = 10,
                                                 class_mode = 'categorical',
                                                 color_mode = 'grayscale')

generate_validation_data = ImageDataGenerator(rescale = 1./255)

validation_dataset = generate_validation_data.flow_from_directory(r'C:\Users\pa662\PycharmProjects\HGM\V11\datasets\validation',
                                                 target_size = (64, 64),
                                                 batch_size = 10,
                                                 class_mode = 'categorical',
                                                 color_mode = 'grayscale')

## Start training
X = train_dataset
Y = validation_dataset
Z = model.fit_generator(
    X,
    steps_per_epoch=len(X),
    epochs=5,
    validation_data = Y,
    validation_steps =len(Y)
)

##Plotting of accuracy
plot.title('Accuracy of the training model')
plot.plot(Z.history['accuracy'], label='train accuracy')
plot.plot(Z.history['val_accuracy'], label='validation accuracy')
plot.xlabel('Number of epochs')
plot.ylabel('Accuracy')
plot.legend()
plot.show()
plot.savefig('Accuracy')

#Plotting of loss
plot.title('Loss of the training model')
plot.plot(Z.history['loss'], label='train loss')
plot.plot(Z.history['val_loss'], label='validation loss')
plot.xlabel('Number of epochs')
plot.ylabel('Loss')
plot.legend()
plot.show()
plot.savefig('Loss')
# Saving the model
model.save("gestpred2.h5")

loss, accuracy = model.evaluate(Y)
