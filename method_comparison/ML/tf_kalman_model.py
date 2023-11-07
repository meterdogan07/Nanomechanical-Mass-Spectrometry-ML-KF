# -*- coding: utf-8 -*-
"""kalman_model.ipynb
"""
import sys
import os
import numpy as np
import scipy.io
import pandas as pd
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer, Input, BatchNormalization, LayerNormalization
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler


# read data into memory
label = y
window_size = M
X = np.asarray(X)
label = np.asarray(label)
label = label.flatten()

print(M)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.01)
X_train, y_train = X, label

data = pd.DataFrame({"X0":X[:, 0], "X1":X[:, 1], "label":label.T})
data.loc[data["label"] == 1]

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.10
    epochs_drop = 50.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(200, activation="tanh"))
model.add(tf.keras.layers.Dense(100, activation="tanh"))
model.add(tf.keras.layers.Dense(50, activation="tanh"))
model.add(tf.keras.layers.Dense(10, activation="tanh"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(loss = tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000001), metrics =["accuracy"])
model.fit(X_train, y_train, epochs=250, batch_size=2000, callbacks=callbacks_list, verbose=1)

model_savedir = "./ML/saved_models/kalman_tf_"+str(int(window_size))+".h5"
print(model_savedir)
model.save(model_savedir)