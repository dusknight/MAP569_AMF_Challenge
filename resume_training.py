import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.core import Dropout
import pandas as pd
from keras import optimizers
from helper import read_x_train, fill_nan
from attention import Attention
from helper import simple_split, dataset_preparation_for_rnn, limit_max_length, rebalance_data
from keras import backend as K

FEATURE_DIM = 37
EPOCH = 70
LR = 5e-5
MAX_LEN = 32
STEPS_PER_EPOCH = 2000
MODEL_WEIGHT_PATH = 'model/'
ind2type = {0: 'HFT', 1: 'MIX', 2: 'NON HFT'}

x_test = read_x_train('data/AMF_test_X.csv',
                      includeShare=True, includeDay=True)
x_test = fill_nan(x_test)

trader_x_test_list = x_test['Trader'].unique()
trader_x_test_list.sort()
trader_x_test_data = [np.array(x_test[x_test['Trader'] == trader].sort_values(
    by=['Day']).drop(['Trader'], axis=1)) for trader in trader_x_test_list]


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


class HistoryCheck(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.f1_m = []
        self.val_f1_m = []

    def on_epoch_end(self, epoch, logs={}):
        self.f1_m.append(logs.get('f1_m'))
        self.val_f1_m.append(logs.get('val_f1_m'))
        if logs.get('f1_m') > 0.9 and logs.get('val_f1_m') > 0.9:
            self.model.save_weights('model/model_weight_'+str(epoch)+'.h5')
            res = []
            for i, trader in enumerate(trader_x_test_list):
                x_test = trader_x_test_data[i].reshape(1, -1, FEATURE_DIM)
                augmented_trader_data = []
                if x_test[0].shape[0] > MAX_LEN:
                    p = 0
                    while p + MAX_LEN < x_test[0].shape[0]:
                        augmented_trader_data.append(
                            x_test[0][p:p+MAX_LEN, :])
                        p += MAX_LEN // 2
                    augmented_trader_data.append(x_test[0][p:, :])
                else:
                    augmented_trader_data.append(x_test[0])
                this_ress = np.array([np.argmax(self.model.predict(data.reshape(
                    1, -1, FEATURE_DIM)), axis=-1)[0] for data in augmented_trader_data])
                this_res = 0 if len(np.where(this_ress == 0)[0])/len(this_ress) >= 0.85 else (
                    1 if len(np.where(this_ress == 1)[0])/len(this_ress) >= 0.5 else 2)
                res.append(this_res)
            output_name = str(epoch)
            with open('output/' + output_name + '.csv', 'w', encoding='utf-8') as f:
                f.write('Trader,type')
                f.write('\n')
                for i, r in enumerate(res):
                    f.write(trader_x_test_list[i] + ',' + ind2type[r] + '\n')
                f.close()

    def on_train_end(self, logs={}):
        output_name = 'log'
        with open('output/' + output_name + '.csv', 'w', encoding='utf-8') as f:
            f.write(str(self.f1_m))
            f.write(str(self.val_f1_m))


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

trader_train_data, label_train_data = dataset_preparation_for_rnn(
    x_train, y_train)

trader_train_data, label_train_data = limit_max_length(
    trader_train_data, label_train_data)
trader_train_data, label_train_data = rebalance_data(
    trader_train_data, label_train_data)

trader_train_data, label_train_data, trader_test_data, label_test_data = simple_split(
    trader_train_data, label_train_data)


print(len(label_train_data))
print(len(label_test_data))
model = keras.Sequential()

model.add(layers.Bidirectional(layers.LSTM(
    32, return_sequences=True, input_shape=(None, FEATURE_DIM)), input_shape=(None, FEATURE_DIM)))
model.add(layers.LayerNormalization())
model.add(Attention(name='attention_weight'))
model.add(Dropout(0.2))
model.add(layers.Dense(32, activation='elu'))
model.add(layers.LayerNormalization())
model.add(Dropout(0.2))
model.add(layers.Dense(10, activation='elu'))
model.add(layers.LayerNormalization())
model.add(Dropout(0.2))
model.add(layers.Dense(3, activation='softmax'))

model.summary()

historycheck = HistoryCheck()
opt = optimizers.Adam(lr=LR)
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy', f1_m])

model.load_weights(MODEL_WEIGHT_PATH)
model.fit(train_generator(trader_train_data, label_train_data), steps_per_epoch=STEPS_PER_EPOCH, validation_steps=STEPS_PER_EPOCH / 10,
          epochs=EPOCH, verbose=1, validation_data=train_generator(trader_test_data, label_test_data), callbacks=[historycheck])

model.save_weights('model/model_weight_final.h5')
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
    this_ress = np.array([np.argmax(model.predict(data.reshape(
        1, -1, FEATURE_DIM)), axis=-1)[0] for data in augmented_trader_data])
    this_res = 0 if len(np.where(this_ress == 0)[0])/len(this_ress) >= 0.85 else (
        1 if len(np.where(this_ress == 1)[0])/len(this_ress) >= 0.5 else 2)
    res.append(this_res)

with open('output/final_round.csv', 'w', encoding='utf-8') as f:
    f.write('Trader,type')
    f.write('\n')
    for i, r in enumerate(res):
        f.write(trader_x_test_list[i] + ',' + ind2type[r] + '\n')
    f.close()
