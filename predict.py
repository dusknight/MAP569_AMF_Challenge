import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.core import Dropout
import pandas as pd
from keras import optimizers
from helper import read_x_train, fill_nan
from attention import Attention
from helper import simple_split, dataset_preparation_for_rnn, limit_max_length, rebalance_data, padding_with_0, f1_m
from keras import backend as K

FEATURE_DIM = 37
EPOCH = 70
LR = 5e-5
MAX_LEN = 16
STEPS_PER_EPOCH = 2000
MODEL_WEIGHT_PATH = 'model/model_weight_final_multihead.h5'
ind2type = {0: 'HFT', 1: 'MIX', 2: 'NON HFT'}

x_test = read_x_train('data/AMF_test_X.csv',
                      includeShare=True, includeDay=True)
x_test = fill_nan(x_test)

trader_x_test_list = x_test['Trader'].unique()
trader_x_test_list.sort()
trader_x_test_data = [np.array(x_test[x_test['Trader'] == trader].sort_values(
    by=['Day']).drop(['Trader'], axis=1)) for trader in trader_x_test_list]


inputs = keras.Input(shape=(None, FEATURE_DIM))
biDirectLSTMlayer = layers.Bidirectional(layers.LSTM(
    32, return_sequences=True))(inputs)

attentionlayer = layers.MultiHeadAttention(
    num_heads=8, key_dim=64, attention_axes=(1, 2))(biDirectLSTMlayer, biDirectLSTMlayer)

dropout_1 = layers.Dropout(0.3)(attentionlayer)

denselayer_1 = layers.Dense(32, activation='elu')(dropout_1)

poolling = layers.GlobalMaxPooling1D()(denselayer_1)

normalize_1 = layers.LayerNormalization()(poolling)

dropout_2 = layers.Dropout(0.3)(normalize_1)

denselayer_2 = layers.Dense(16, activation='elu')(dropout_2)

normalize_2 = layers.LayerNormalization()(denselayer_2)

outputs = layers.Dense(3, activation='softmax')(normalize_2)

model = keras.Model(inputs, outputs, name="rnn")

model.summary()

opt = optimizers.Adam(lr=LR)
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy', f1_m])

model.load_weights(MODEL_WEIGHT_PATH)

####################################################
# Evaluate
####################################################
res = []
for i, trader in enumerate(trader_x_test_list):
    x_test = trader_x_test_data[i].reshape(1, -1, FEATURE_DIM)
    augmented_trader_data = []
    if x_test[0].shape[0] > MAX_LEN:
        p = 0
        while p + MAX_LEN < x_test[0].shape[0]:
            augmented_trader_data.append(x_test[0][p:p+MAX_LEN, :])
            p += MAX_LEN // 2
        augmented_trader_data.append(x_test[0][p:, :])
    else:
        augmented_trader_data.append(x_test[0])
    augmented_trader_data = np.array(padding_with_0(
        augmented_trader_data, max_length=MAX_LEN)).reshape(-1, MAX_LEN, FEATURE_DIM)
    this_ress = np.argmax(model.predict(augmented_trader_data), axis=1)
    this_res = 0 if len(np.where(this_ress == 0)[0])/len(this_ress) >= 0.85 else (
        1 if len(np.where(this_ress == 1)[0])/len(this_ress) >= 0.5 else 2)
    res.append(this_res)

with open('output/final_round_maxpooling.csv', 'w', encoding='utf-8') as f:
    f.write('Trader,type')
    f.write('\n')
    for i, r in enumerate(res):
        f.write(trader_x_test_list[i] + ',' + ind2type[r] + '\n')
    f.close()
