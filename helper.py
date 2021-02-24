import pandas as pd
import numpy as np
from collections import Counter
import random
import math
from keras import backend as K
from sklearn import preprocessing


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


def read_x_train(filename: str, includeTrader=True, includeShare=True, includeDay=True) -> pd.DataFrame:
    """Read a csv file and do a little bit conversion from string to int
    Args
        filename: csv file name
        includeTrader: if the column 'Trader' is included
        includeShare: if the column 'Share' is included
        includeDay: if the column 'Day' is included

    Returns
        x_remove: pd.DataFrame
            A dataframe after cleaning
    """
    x_remove = pd.read_csv(filename)
    x_remove['Day'] = x_remove['Day'].apply(lambda x: int(x.split('_')[1]))
    # x_remove['Trader'] = x_remove['Trader'].apply(
    #     lambda x: int(x.split('_')[1]))
    x_remove['Share'] = x_remove['Share'].apply(
        lambda x: int(x.split('_')[1]))
    if not includeTrader:
        x_remove = x_remove.drop(
            ['Trader'], axis=1, inplace=False)
    if not includeShare:
        x_remove = x_remove.drop(
            ['Share'], axis=1, inplace=False)
    if not includeDay:
        x_remove = x_remove.drop(
            ['Day'], axis=1, inplace=False)
    x_remove = x_remove.drop(
        ['Index'], axis=1, inplace=False)
    return x_remove


def fill_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Deal with the NaN value in our specific dataFrame. 
       Will simply fill the NaN in OMR by 0. For the NaN concerning the dt,
       will do the conversion x -> 1/x then fill the NaN by 0.
    Args
        df: pd.DataFrame

    Returns
        df: pd.DataFrame
            A dataframe after cleaning
    """
    dt_list = ['min_dt_TV1', 'mean_dt_TV1', 'med_dt_TV1', 'min_dt_TV1_TV2',
               'mean_dt_TV1_TV2', 'med_dt_TV1_TV2', 'min_dt_TV1_TV3',
               'mean_dt_TV1_TV3', 'med_dt_TV1_TV3', 'min_dt_TV1_TV4',
               'mean_dt_TV1_TV4', 'med_dt_TV1_TV4']
    df['OMR'] = df['OMR'].fillna(0)
    df['OTR'] = df['OTR'].fillna(0)
    for col_name in dt_list:
        df[col_name] = df[col_name].apply(
            lambda x: 1/x if pd.notnull(x) else 0)
    return df


def standardization(train_df: pd.DataFrame, test_df: pd.DataFrame):
    not_scale_list = ['Share', 'Day', 'Trader']
    for col_name in train_df.columns:
        if col_name not in not_scale_list:
            combined = preprocessing.scale(
                train_df[col_name].append(test_df[col_name]))
            train_df[col_name] = combined[:len(train_df[col_name])]
            test_df[col_name] = combined[len(train_df[col_name]):]
    return train_df, test_df


def dataset_preparation_for_rnn(x_data: pd.DataFrame, y_data: pd.DataFrame):
    """Group by 'Trader' and sort according to the 'Day'. Then for each 'Trader'
       construct a list of vector. The y_data is encoding in one-hot.
    Args
        x_data: pd.DataFrame
        y_data: pd.DataFrame
    Returns
        trader_data: list[np.array]
            A list of each trader's events
        label_data: list[list]
            One hot encoding of 3 classes
    """
    trader_list = x_data['Trader'].unique()
    trader_list.sort()
    trader_data = [np.array(x_data[x_data['Trader'] == trader].sort_values(
        by=['Day']).drop(['Trader'], axis=1)) for trader in trader_list]
    label_data = list(y_data.sort_values(by=['Trader'])['type'].apply(
        lambda x: [1, 0, 0] if x == 'HFT' else ([0, 1, 0] if x == 'MIX' else [0, 0, 1])))

    return trader_data, label_data


def simple_split(trader_data: list, label_data: list, ratio=0.1):
    """First shuffle the data randomly then split train/test data by ratio.
       The split is trader-wise.
    Args
        trader_data: list
        label_data: list
    Returns
        trader_train_data, label_train_data, trader_test_data, label_test_data
    """
    # split

    c = list(zip(trader_data, label_data))
    random.shuffle(c)
    trader_data, label_data = zip(*c)
    n = len(label_data)
    test_n = math.floor(n * ratio)
    trader_test_data = trader_data[:test_n]
    label_test_data = label_data[:test_n]
    trader_train_data = trader_data[test_n:]
    label_train_data = label_data[test_n:]
    return trader_train_data, label_train_data, trader_test_data, label_test_data


def limit_max_length(trader_data: list, label_data: list, max_length=32):
    """Cut the series of trader_data into many sub-series of max_length, with overlapping 
       max_length / 2. For example, [1, 0, 2, 3, 5, 1] with a max_length=2 will be transformed
       into [1, 0] [0, 2] [2, 3] [3, 5] [5, 1]
    Args
        trader_data: list
        label_data: list
        max_length: int
            The maximum window size of sub-series
    Returns
        augmented_trader_data, augmented_label_data
    """
    augmented_trader_data = []
    augmented_label_data = []
    for i, data in enumerate(trader_data):
        if data.shape[0] > max_length:
            p = 0
            while p + max_length < data.shape[0]:
                augmented_trader_data.append(data[p:p+max_length, :])
                augmented_label_data.append(label_data[i])
                p += max_length // 2
            augmented_trader_data.append(data[p:, :])
            augmented_label_data.append(label_data[i])
        else:
            augmented_trader_data.append(data)
            augmented_label_data.append(label_data[i])

    return augmented_trader_data, augmented_label_data


def rebalance_data(trader_data: list, label_data: list, max_length=32):
    """Resampling the minority classes, the resampled data are of max_length.
    Args
        trader_data: list
        label_data: list
        max_length: int
            The maximum window size of sub-series
    Returns
        trader_data, label_data
    """
    c = Counter([tuple(data) for data in label_data])
    most_common, max_n = c.most_common(1)[0]
    possible_labels = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    rng = np.random.default_rng()
    for possible_label in possible_labels:
        if tuple(possible_label) == most_common:
            continue
        inds_for_this_labels = [i for i, label in enumerate(
            label_data) if label == possible_label]
        all_events = np.vstack([trader_data[i] for i in inds_for_this_labels])
        for i in range(max_n-c[tuple(possible_label)]):
            artificial_event = rng.choice(all_events, max_length)
            artificial_event = artificial_event[np.argsort(
                artificial_event[:, 1])]
            trader_data.append(artificial_event)
            label_data.append(possible_label)

    return trader_data, label_data


def padding_with_0(trader_data: list, max_length=32):
    """Padding the data which is not of max_length with 0-rows.
    Args
        trader_data: list
        max_length: int
            The maximum window size of sub-series
    Returns
        trader_data
    """
    for i, data in enumerate(trader_data):
        if data.shape[0] < max_length:
            trader_data[i] = np.pad(
                data, ((0, max_length - data.shape[0]), (0, 0)), 'constant')
    return trader_data
# %%


def trans2trader(x_data, y_pred):
    res = list(zip(x_data.Trader, y_pred))
    # res = [(row.Trader, y_pred[i])  for i, row in x_data.iterrows()]

    dicts = {}
    for row in res:
        if row[0] in dicts:
            dicts[row[0]].append(row[1])
        else:
            dicts[row[0]] = [row[1]]

    final = {}
    for k, v in dicts.items():
        if len(np.where(np.array(v) == 0)[0])/len(v) > 0.85:
            final[k] = 'HFT'
        elif len(np.where(np.array(v) == 1)[0])/len(v) > 0.5:
            final[k] = 'MIX'
        else:
            final[k] = 'NON HFT'
    return final
