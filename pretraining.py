from re import X
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.core import Dropout
import pandas as pd
from keras import optimizers
from helper import padding_with_0, preprocessing_for_pretraining, read_x_train, fill_nan
from helper import simple_split, dataset_preparation_for_rnn, limit_max_length, rebalance_data, f1_m, standardization
from keras.callbacks import ReduceLROnPlateau
from config import *

ind2type = {0: 'HFT', 1: 'MIX', 2: 'NON HFT'}

x_test = read_x_train('data/AMF_test_X.csv',
                      includeShare=True, includeDay=True)
x_test = fill_nan(x_test)
x_train = read_x_train('data/AMF_train_X.csv',
                       includeShare=True, includeDay=True)
x_train = fill_nan(x_train)

train, train_label, valid, valid_label = preprocessing_for_pretraining(
    x_train, x_test, split_valid_set=True)

model = build_pretraining_model()
reduce_lr_on_plateau = ReduceLROnPlateau(
    monitor='mean_squared_error', factor=0.5, patience=5, verbose=1, mode='auto', cooldown=0, min_lr=1e-6)
opt = optimizers.Adam(lr=LR)
model.compile(loss='mean_squared_error',
              optimizer=opt, metrics=[tf.keras.metrics.MeanSquaredError()])
# model.fit(train_generator(trader_train_data, label_train_data), steps_per_epoch=STEPS_PER_EPOCH, validation_steps=STEPS_PER_EPOCH / 10,
#           epochs=EPOCH, verbose=1, validation_data=train_generator(trader_test_data, label_test_data), callbacks=[historycheck])

model.fit(train, train_label, batch_size=BATCH_SIZE,
          epochs=EPOCH, verbose=1, validation_data=(valid, valid_label), callbacks=[reduce_lr_on_plateau])

model.save_weights('model/model_weight_pretrain.h5')
