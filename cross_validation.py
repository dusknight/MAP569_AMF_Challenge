from re import X
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.core import Dropout
import pandas as pd
from keras import optimizers
from helper import padding_with_0, preprocessing_for_cross_valid, read_x_train, fill_nan
from helper import simple_split, dataset_preparation_for_rnn, limit_max_length, rebalance_data, f1_m, standardization
from keras.callbacks import ReduceLROnPlateau
from config import *
import pickle

ind2type = {0: 'HFT', 1: 'MIX', 2: 'NON HFT'}

x_test = read_x_train('data/AMF_test_X.csv',
                      includeShare=True, includeDay=True)
x_test = fill_nan(x_test)
x_train = read_x_train('data/AMF_train_X.csv',
                       includeShare=True, includeDay=True)
x_train = fill_nan(x_train)
x_train, x_test = standardization(x_train, x_test)

trader_x_test_list = x_test['Trader'].unique()
trader_x_test_list.sort()
trader_x_test_data = [np.array(x_test[x_test['Trader'] == trader].sort_values(
    by=['Day']).drop(['Trader'], axis=1)) for trader in trader_x_test_list]

y_train = pd.read_csv('data/AMF_train_Y.csv')

folds_train_data = pickle.load(open('data/cross_valid_data.obj', "rb"))


# class HistoryCheck(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.f1_m = []
#         self.val_f1_m = []

#     def on_epoch_end(self, epoch, logs={}):
#         self.f1_m.append(logs.get('f1_m'))
#         self.val_f1_m.append(logs.get('val_f1_m'))
#         if logs.get('f1_m') > 0.98 and logs.get('val_f1_m') > 0.98:
#             self.model.save_weights('model/model_weight_'+str(epoch)+'.h5')

#     def on_train_end(self, logs={}):
#         output_name = 'log'
#         with open('output/' + output_name + '.csv', 'w', encoding='utf-8') as f:
#             for i, v in enumerate(self.f1_m):
#                 f.write(str(v) + ', ' + str(self.val_f1_m[i]) + '\n')


# for data in folds_train_data:
#     model = bulid_model()
#     reduce_lr_on_plateau = ReduceLROnPlateau(
#         monitor='val_f1_m', factor=0.5, patience=10, verbose=1, mode='auto', cooldown=5, min_lr=1e-6)
#     historycheck = HistoryCheck()
#     opt = optimizers.Adam(lr=LR)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=opt, metrics=['accuracy', f1_m])
#     model.fit(data[0], data[1], batch_size=BATCH_SIZE,
#               epochs=CROSSVALID_EPOCH, verbose=1, validation_data=(data[2], data[3]), callbacks=[historycheck, reduce_lr_on_plateau])

# model.save('model/model_weight_final_crossvalid.h5')

# ####################################################
# # Evaluate
# ####################################################
# res = []
# for i, trader in enumerate(trader_x_test_list):
#     x_test = trader_x_test_data[i].reshape(1, -1, FEATURE_DIM)
#     augmented_trader_data = []
#     if x_test[0].shape[0] > MAX_LEN:
#         p = 0
#         while p + MAX_LEN < x_test[0].shape[0]:
#             augmented_trader_data.append(x_test[0][p:p+MAX_LEN, :])
#             p += MAX_LEN // 2
#         augmented_trader_data.append(x_test[0][p:, :])
#     else:
#         augmented_trader_data.append(x_test[0])
#     augmented_trader_data = np.array(padding_with_0(
#         augmented_trader_data, max_length=MAX_LEN)).reshape(-1, MAX_LEN, FEATURE_DIM)

#     this_ress = np.argmax(model.predict(augmented_trader_data), axis=1)
#     this_res = 0 if len(np.where(this_ress == 0)[0])/len(this_ress) >= 0.85 else (
#         1 if len(np.where(this_ress == 1)[0])/len(this_ress) >= 0.5 else 2)
#     res.append(this_res)

# with open('output/final_round_crossvalid.csv', 'w', encoding='utf-8') as f:
#     f.write('Trader,type')
#     f.write('\n')
#     for i, r in enumerate(res):
#         f.write(trader_x_test_list[i] + ',' + ind2type[r] + '\n')
#     f.close()
