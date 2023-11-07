import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import keras
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

M = int(M)
window_size = M
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
print(np.shape(X_train))
print(np.shape(y_train2))
print(np.sum(y_train))

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.01)
X_train, y_train = X, events

# fit model no training data
model = xgb.XGBClassifier(max_depth=10, use_label_encoder=False, verbose = 0)
model.fit(X_train, y_train, verbose = False)

model_savedir = "./Blackbox_ML/classification_xgbmodel"+str(int(window_size))
print(model_savedir)
joblib.dump(model,model_savedir)


multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(verbosity=2,max_depth=10,objective='reg:squarederror'))
multioutputregressor.fit(X2_train, y_train2, verbose = False)

model_savedir2 = "./Blackbox_ML/regression_xgbmodel"+str(int(window_size))
print(model_savedir)
joblib.dump(multioutputregressor,model_savedir2)