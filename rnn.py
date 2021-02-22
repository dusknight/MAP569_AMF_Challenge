import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.core import Dropout
import pandas as pd
from keras import optimizers
from helper import read_x_train, fill_nan
from attention import Attention

FEATURE_DIM = 37
EPOCH = 75


def dataset_preparation(x_data: pd.DataFrame, y_data: pd.DataFrame):
    trader_list = x_data['Trader'].unique()
    trader_list.sort()
    trader_data = [np.array(x_data[x_data['Trader'] == trader].sort_values(
        by=['Day']).drop(['Trader'], axis=1)) for trader in trader_list]
    label_data = list(y_data.sort_values(by=['Trader'])['type'].apply(
        lambda x: [1, 0, 0] if x == 'HFT' else ([0, 1, 0] if x == 'MIX' else [0, 0, 1])))

    return trader_data, label_data


def simple_split(trader_data: list, label_data: list, ratio=0.1):
    # split
    import math
    n = len(label_data)
    test_n = math.floor(n * ratio)
    trader_test_data = trader_data[:test_n]
    label_test_data = label_data[:test_n]
    trader_train_data = trader_data[test_n:]
    label_train_data = label_data[test_n:]
    return trader_train_data, label_train_data, trader_test_data, label_test_data


def train_generator(trader_data, label_data):
    while True:
        rand_ind = np.random.randint(len(label_data))
        x_train = trader_data[rand_ind].reshape(1, -1, FEATURE_DIM)
        # y_train will depend on past 5 timesteps of x
        y_train = np.array(label_data[rand_ind]).reshape(1, -1)
        yield x_train, y_train


x_train = read_x_train('data/AMF_train_X.csv',
                       includeShare=True, includeDay=True)
x_train = fill_nan(x_train)
y_train = pd.read_csv('data/AMF_train_Y.csv')

trader_train_data, label_train_data = dataset_preparation(x_train, y_train)
trader_train_data, label_train_data, trader_test_data, label_test_data = simple_split(
    trader_train_data, label_train_data)

model = keras.Sequential()

model.add(layers.Bidirectional(layers.LSTM(
    64, return_sequences=True, input_shape=(None, FEATURE_DIM)), input_shape=(None, FEATURE_DIM)))
model.add(layers.LayerNormalization())
model.add(Attention(name='attention_weight'))
model.add(Dropout(0.2))
model.add(Attention(name='attention_weight'))
model.add(Dropout(0.2))
model.add(layers.Dense(40, activation='elu'))
model.add(layers.LayerNormalization())
model.add(Dropout(0.2))
model.add(layers.Dense(10, activation='elu'))
model.add(layers.Dense(3, activation='softmax'))

model.summary()

opt = optimizers.Adam(lr=3e-5)
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=["accuracy"])

model.fit(train_generator(trader_train_data, label_train_data), steps_per_epoch=70, validation_steps=10,
          epochs=EPOCH, verbose=1, validation_data=train_generator(trader_test_data, label_test_data))

x_data = read_x_train('data/AMF_test_X.csv',
                      includeShare=True, includeDay=True)
x_data = fill_nan(x_data)

trader_list = x_data['Trader'].unique()
trader_list.sort()
trader_data = [np.array(x_data[x_data['Trader'] == trader].sort_values(
    by=['Day']).drop(['Trader'], axis=1)) for trader in trader_list]

res = []
for i, trader in enumerate(trader_list):
    x_train = trader_data[i].reshape(1, -1, FEATURE_DIM)
    res.append(np.argmax(model.predict(x_train), axis=-1))

ind2type = {0: 'HFT', 1: 'MIX', 2: 'NON HFT'}
with open('output/attention.csv', 'w', encoding='utf-8') as f:
    f.write('Trader,type')
    f.write('\n')
    for i, r in enumerate(res):
        f.write(trader_list[i] + ',' + ind2type[r[0]] + '\n')
    f.close()
