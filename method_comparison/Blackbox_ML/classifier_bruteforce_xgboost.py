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
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

window_size = M
model_savedir = "./Blackbox_ML/classification_xgbmodel"+str(int(window_size))

model_xgb = joblib.load(model_savedir)

event = model_xgb.predict(x)



