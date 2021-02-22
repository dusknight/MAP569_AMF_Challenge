from tensorflow import keras
from helper import read_x_train, fill_nan
import numpy as np

FEATURE_DIM = 35
model = keras.models.load_model('model/attention')

x_data = read_x_train('data/AMF_test_X.csv',
                      includeShare=True, includeDay=True)
trader_list = x_data['Trader'].unique()
trader_list.sort()
trader_data = [np.array(x_data[x_data['Trader'] == trader].sort_values(
    by=['Day']).drop(['Trader', 'Share', 'Day'], axis=1)) for trader in trader_list]

res = []
for i, trader in enumerate(trader_list):
    x_train = trader.reshape(1, -1, FEATURE_DIM)
    res.append(model.predict_classes(x_train))
