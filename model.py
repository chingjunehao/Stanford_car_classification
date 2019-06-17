# -*- coding: utf-8 -*-
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
import cv2

import os
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.densenet import DenseNet201

from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Concatenate, AveragePooling2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from numpy.random import seed
seed(2019)
from tensorflow import set_random_seed
set_random_seed(2019)


curr_dir = os.getcwd()
train_dir = os.path.join(curr_dir,'train')


NUM_EPOCHS = 50
BS = 24
IMG_WIDTH=224
IMG_HEIGHT=224


train_datagen = ImageDataGenerator(rescale=1 / 255.0,
rotation_range=20,
zoom_range=0.2,
width_shift_range=0.1,
 height_shift_range=0.1,
horizontal_flip=True,
 validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BS,
    class_mode='categorical',
    subset='training', shuffle=True) # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_dir, # same directory as training data
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BS,
    class_mode='categorical',
    subset='validation') # set as validation data

base_model = DenseNet201(include_top=False, weights='imagenet',  input_shape=(IMG_WIDTH,IMG_HEIGHT,3))

x = base_model.output
x = AveragePooling2D((7, 7))(x)
x = Flatten()(x)

concat1 = GlobalAveragePooling2D()(base_model.get_layer("conv5_block22_concat").output)
x = Concatenate()([x, concat1])

predictions = Dense(196, activation='softmax')(x)
model = Model(inputs=base_model.inputs, outputs=predictions)

opt = Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

model.summary()

tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
log_file_path = 'logs/training.log'
early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1)
trained_models_path = 'models/model'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.h5'
model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)

model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BS,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // BS,
    epochs = NUM_EPOCHS, callbacks=[tensor_board, model_checkpoint, early_stop]
)
