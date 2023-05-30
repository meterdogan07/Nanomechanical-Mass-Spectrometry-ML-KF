import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer, Input, BatchNormalization, LayerNormalization
import keras
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler

M = int(M)
dataset = np.asarray(dataset)
dataset2 = np.asarray(dataset2)
X = dataset[:,0:M]
X2 = dataset2[:,0:M]

events = dataset[:,M+2]
locations = dataset2[:,M]
dys = dataset2[:,M+1]

X_train = np.array(X)
X2_train = np.array(X2)
y_train = events
y_train2 = np.vstack([100000*dys, locations]).T

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.50
    epochs_drop = 25.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# define the keras model
model = Sequential()
model.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='relu', input_shape=(M,1)))
model.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='relu', input_shape=(M,1)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='relu', input_shape=(M,1)))
model.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='relu', input_shape=(M,1)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='relu', input_shape=(M,1)))
model.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='relu', input_shape=(M,1)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(Dense(50, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(20, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000001),
              loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives()])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=200, batch_size=10, callbacks=callbacks_list, verbose=1)

model_savedir = "./Blackbox_ML/classification_cnn_"+str(int(M))+".h5"
print(model_savedir)
model.save(model_savedir)

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.50
    epochs_drop = 25.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# define the keras model
model2 = Sequential()
model2.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='relu', input_shape=(M,1)))
model2.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='relu', input_shape=(M,1)))
model2.add(layers.MaxPooling1D(pool_size=2))
model2.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='relu', input_shape=(M,1)))
model2.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='relu', input_shape=(M,1)))
model2.add(layers.MaxPooling1D(pool_size=2))
model2.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='relu', input_shape=(M,1)))
model2.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='relu', input_shape=(M,1)))
model2.add(layers.MaxPooling1D(pool_size=2))
model2.add(layers.Flatten())
model2.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
model2.add(Dense(20, activation='relu', kernel_initializer='he_normal'))
model2.add(Dense(2, activation='linear', kernel_initializer='he_normal'))

# compile the keras model
model2.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['mse', 'mae'])

# fit the keras model on the dataset
model2.fit(X2_train, y_train2, epochs=200, batch_size=20, callbacks=callbacks_list, verbose=1)

model2_savedir = "./Blackbox_ML/regression_cnn_"+str(int(M))+".h5"
print(model2_savedir)
model2.save(model2_savedir)