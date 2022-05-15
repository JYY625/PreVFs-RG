from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.layers import Flatten
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Conv1D, MaxPooling1D,ReLU,BatchNormalization,Activation

inputs = Input(shape=(1,1282))
layer0 = Conv1D(filters=64, kernel_size=3, padding='same',activation='relu')(inputs)
layer1 = Conv1D(filters=128, kernel_size=3, padding='same',activation='relu')(layer0)
layer2 = Conv1D(filters=64, kernel_size=3, padding='same',activation='relu')(layer1)
skip0 = Activation("relu")(layer0)
skip0 = BatchNormalization()(skip0)
skip0 = Activation('relu')(skip0)
layer3 = layer2+layer0
layer4 = Conv1D(filters=256, kernel_size=3,padding='same',activation='relu')(layer3)
layer5 = Conv1D(filters=128, kernel_size=3, padding='same',activation='relu')(layer4)
skip = Conv1D(filters=128, kernel_size=1,padding='same',activation='relu')(layer3)
skip = BatchNormalization()(skip)
skip = Activation('relu')(skip)
layer6 = skip+layer5
layer7 = GRU(128, return_sequences=True)(layer6)
layer9 = Flatten()(layer7)
layer10 = Dense(64, activation='relu')(layer9)
layer11 = Dense(32, activation='relu')(layer10)
outputs = tf.keras.layers. Dense(2,activation="softmax")(layer11)






