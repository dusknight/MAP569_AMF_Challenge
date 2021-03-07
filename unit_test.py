from re import L
from helper import read_x_train, fill_nan, dataset_preparation_for_rnn, limit_max_length, rebalance_data, padding_with_0
import pandas as pd
from collections import Counter


def test_limit_max_length():
    x_train = read_x_train('data/AMF_train_X.csv',
                           includeShare=True, includeDay=True)
    x_train = fill_nan(x_train)
    y_train = pd.read_csv('data/AMF_train_Y.csv')

    trader_train_data, label_train_data = dataset_preparation_for_rnn(
        x_train, y_train)

    trader_train_data, label_train_data = limit_max_length(
        trader_train_data, label_train_data)
    print(len(trader_train_data))

    print(Counter([tuple(data) for data in label_train_data]))
    # for data in trader_train_data:
    #     print(data.shape)


def test_rebalance():
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
    print(Counter([tuple(data) for data in label_train_data]))


def test_padding():
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
    trader_train_data = padding_with_0(trader_train_data)
    length_set = set([data.shape for data in trader_train_data])
    print(length_set)


def test_split():
    pass


test_padding()
