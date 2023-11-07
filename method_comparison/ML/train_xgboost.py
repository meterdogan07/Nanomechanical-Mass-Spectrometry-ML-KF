# -*- coding: utf-8 -*-
"""kalman_model.ipynb
"""
import sys
import os
import joblib
import numpy as np
import scipy.io
import pandas as pd
import math
import tensorflow as tf
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer, Input, BatchNormalization, LayerNormalization
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler


# read data into memory
#X = sys.argv[1]
#label = sys.argv[2]
#window_size = sys.argv[1]
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

# fit model no training data
model = xgb.XGBClassifier(max_depth=11, use_label_encoder=False, verbose = 0)
model.fit(X_train, y_train, verbose = True)

model_savedir = "./ML/saved_models/xgbmodel"+str(int(window_size))
print(model_savedir)
joblib.dump(model,model_savedir)