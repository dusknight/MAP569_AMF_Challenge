from collections import Counter

import numpy as np
import pandas as pd

from helper import read_x_train

x_train = read_x_train('data/AMF_train_X.csv',
                       includeShare=True, includeDay=True)
x_test = read_x_train('data/AMF_test_X.csv',
                      includeShare=True, includeDay=True)
y_train = pd.read_csv('data/AMF_train_Y.csv')
trader_train = y_train['type']
c = Counter(y_train['type'])
default_ratio = [e[1] for e in sorted(
    [(i, c[i] / len(trader_train)) for i in c], key=lambda elem: elem[0])]
name2type = dict(zip(y_train.Trader, y_train.type))


def get_share_trader_distribution(x_train: pd.DataFrame, y_train: pd.DataFrame):
    d = {}
    for share in x_train['Share'].unique():
        type_list = [name2type[trader]
                     for trader in x_train[x_train['Share'] == share]['Trader'].unique()]
        c = Counter(type_list)

        def pairwise(a, b):
            return a / b - 1
        d[share] = list(map(pairwise, [e[1] for e in sorted(
            [(i, c[i] / len(type_list)) for i in c], key=lambda elem: elem[0])], default_ratio))
    return d


def data_augment(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame):
    trader_day_share_count = x_train.groupby(['Trader', 'Day']).Share.agg(
        'count').to_frame('everyday_sharecount').reset_index()
    x_train = x_train.join(trader_day_share_count.set_index(
        ['Trader', 'Day']), on=['Trader', 'Day'])
    trader_day_share_count = x_train.groupby(['Trader', 'Share']).Share.agg(
        'count').to_frame('everyshare_daycount').reset_index()
    x_train = x_train.join(trader_day_share_count.set_index(
        ['Trader', 'Share']), on=['Trader', 'Share'])
    ratio_list = ['share_hft_ratio', 'share_mix_ratio', 'share_non_hft_ratio']
    share_info = pd.DataFrame.from_dict(get_share_trader_distribution(
        x_train, y_train), orient='index', columns=ratio_list)
    share_info['Share'] = share_info.index
    x_train = x_train.join(share_info.set_index(['Share']), on=['Share'])

    trader_day_share_count = x_test.groupby(['Trader', 'Day']).Share.agg(
        'count').to_frame('everyday_sharecount').reset_index()
    x_test = x_test.join(trader_day_share_count.set_index(
        ['Trader', 'Day']), on=['Trader', 'Day'])
    trader_day_share_count = x_test.groupby(['Trader', 'Share']).Share.agg(
        'count').to_frame('everyshare_daycount').reset_index()
    x_test = x_test.join(trader_day_share_count.set_index(
        ['Trader', 'Share']), on=['Trader', 'Share'])

    x_test = x_test.join(share_info.set_index(['Share']), on=['Share'])

    for col in ratio_list:
        x_test[col] = x_test[col].fillna(0)

    x_train['OMR_OTR_ratio'] = x_train['OMR'] / x_train['OTR']
    x_train['OCR_OTR_ratio'] = x_train['OCR'] / x_train['OTR']
    x_train['OCR_OMR_ratio'] = x_train['OCR'] / x_train['OMR']

    x_test['OMR_OTR_ratio'] = x_test['OMR'] / x_test['OTR']
    x_test['OCR_OTR_ratio'] = x_test['OCR'] / x_test['OTR']
    x_test['OCR_OMR_ratio'] = x_test['OCR'] / x_test['OMR']

    return x_train, x_test


if __name__ == "__main__":
    x_train, x_test = data_augment(x_train, y_train, x_test)
    x_train.to_csv('data/Agmt_train_X.csv', index=False)
    x_test.to_csv('data/Agmt_test_X.csv', index=False)
